import os
import tempfile
import shutil
import cv2
import glob
import torch
from collections import OrderedDict
import numpy as np
from cog import BasePredictor, Input, Path
from main_test_swinir import define_model, setup, get_image_pair
from models.network_swinir import SwinIR

# RGB images larger than this throw OOM errors
MAX_PIXELS = 2050000

class Predictor(BasePredictor):
    def setup(self):
        model_dir = "experiments/pretrained_models"

        model_path =  os.path.join(model_dir, "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
        self.scale = 4
        self.window_size = 8
        self.model = SwinIR(upscale=self.scale, in_chans=3, img_size=64, window_size=self.window_size,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                    num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'
        pretrained_model = torch.load(model_path)
        self.model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        self.model.eval()
        self.model.to(self.device)

    def predict(
        self,
        image: Path = Input(
            description="input image",
        )
    ) -> Path:

        # set input folder
        input_dir = "input_cog_temp"

        try:
            os.makedirs(input_dir, exist_ok=True)
            input_path = os.path.join(input_dir, os.path.basename(image))
            shutil.copy(str(image), input_path)

            model = self.model

            window_size = self.window_size
            
            # psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0
            out_path = Path(tempfile.mkdtemp()) / "out.png"

            img_lq = None
            # inference
            with torch.no_grad():

                for _, path in enumerate(sorted(glob.glob(os.path.join(input_dir, "*")))):
                    # read image
                    img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
                    
                    if img_lq.shape[0] * img_lq.shape[1] > MAX_PIXELS and img_lq.shape[2] > 1:
                        raise ValueError("Input RGB image has dimensions width * height > 2.05 million. This will OOM, pass in smaller images")

                    img_lq = np.transpose(
                        img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
                        (2, 0, 1),
                    )  # HCW-BGR to CHW-RGB
                    img_lq = (
                        torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)
                    )  # CHW-RGB to NCHW-RGB
                    # pad input image to be a multiple of window_size
                    _, _, h_old, w_old = img_lq.size()
                    h_pad = (h_old // window_size + 1) * window_size - h_old
                    w_pad = (w_old // window_size + 1) * window_size - w_old
                    img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[
                        :, :, : h_old + h_pad, :
                    ]
                    img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[
                        :, :, :, : w_old + w_pad
                    ]
                    output = model(img_lq)
                    output = output[
                        ..., : h_old * self.scale, : w_old * self.scale
                    ]

                    # save image
                    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    if output.ndim == 3:
                        output = np.transpose(
                            output[[2, 1, 0], :, :], (1, 2, 0)
                        )  # CHW-RGB to HCW-BGR
                    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
                    cv2.imwrite(str(out_path), output)
        except RuntimeError as e:
            if type(e) == torch.cuda.OutOfMemoryError:
                print("""CUDA OOM - this generally happens for RGB images where width * height > 2.05 million pixels""")
                self.setup()
            raise e
        finally:
            shutil.rmtree(input_dir)
        return out_path
