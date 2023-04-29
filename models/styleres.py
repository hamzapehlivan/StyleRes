import torch
import torch.nn as nn
import torch.nn.functional as F
from models.e4e import E4E_Inversion
from models.stylegan2 import Generator
from editings.editor import Editor
from options import Settings

class StyleRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = E4E_Inversion(resolution=256, num_layers = 50, mode='ir_se', out_res=64)
        self.generator = Generator(z_dim=512, w_dim=512, c_dim=0, resolution=1024, img_channels=3, 
                        fused_modconv_default='inference_only', embed_res=64)
        # Set Generator arguments for eval mode
        self.G_kwargs_val = {'noise_mode':'const', 'force_fp32':True}
        self.device = Settings.device
        self.editor = Editor()

    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.encoder.basic_encoder.load_state_dict(ckpt['e4e'], strict=True)
        self.encoder.latent_avg = ckpt['latent_avg']
        self.generator.load_state_dict(ckpt['generator_smooth'], strict=True)
        print("Model succesfully loaded")

    def send_to_device(self):
        self.encoder.to(self.device)
        self.generator.to(self.device)
        if self.device != 'cpu':
            self.encoder.latent_avg = self.encoder.latent_avg.cuda()

    """
        Inputs: Input images and edit configs
        Returns: Edited images
    """
    def edit_images(self, image, cfg):
        image = image.to(self.device)
        with torch.no_grad():
            latents, skips = self.encoder(image)
            input_is_stylespace = True if cfg.method == 'styleclip' and cfg.type == 'global' else False
            if input_is_stylespace:
                latents = self.generator(latents, None, return_styles=True)

        # GradCtrl requires gradients, others do not
        latents_edited = self.editor.edit(latents, cfg) 

        with torch.no_grad():
            # Get F space features F_feats, for the original image
            skips['F_feats'] = self.generator(latents, skips, return_f = True, **self.G_kwargs_val)
            # Transform F_feats to incoming edited image
            images = self.generator(latents_edited, skips, **self.G_kwargs_val)
            
        return images


