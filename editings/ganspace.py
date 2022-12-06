import torch
import csv
from options import Settings
import os

class GanSpace():
    def __init__(self) -> None:

        self.gan_space_configs = {}
        
        with open(os.path.join(Settings.ganspace_directions, 'ganspace_configs.csv'), "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                key = row.pop(0)
                self.gan_space_configs[key] = list(map(int, row))

    def edit(self, latent, cfg):
        with torch.no_grad():
            self.load_ganspace_pca()
            gan_space_config = self.gan_space_configs[cfg.edit]
            gan_space_config[-1] = cfg.strength
            return self.edit_ganspace(latent, gan_space_config)

    def load_ganspace_pca(self):
        try:   # Check if loaded
            getattr(self, f"pca")
        except:
            pca = torch.load(os.path.join(Settings.ganspace_directions, 'ffhq_pca.pt'))
            setattr(self, f"pca", pca)
        

    def edit_ganspace(self, latents, config):
        edit_latents = []
        pca_idx, start, end, strength = config
        for latent in latents:
            delta = self.get_delta( latent, pca_idx, strength)
            delta_padded = torch.zeros(latent.shape).to(Settings.device)
            delta_padded[start:end] += delta.repeat(end - start, 1)
            edit_latents.append(latent + delta_padded)
        return torch.stack(edit_latents)

    def get_delta(self, latent, idx, strength):
        # pca: ganspace checkpoint. latent: (16, 512) w+
        w_centered = latent - self.pca['mean'].to(Settings.device)
        lat_comp = self.pca['comp'].to(Settings.device)
        lat_std = self.pca['std'].to(Settings.device)
        w_coord = torch.sum(w_centered[0].reshape(-1)*lat_comp[idx].reshape(-1)) / lat_std[idx]
        delta = (strength - w_coord)*lat_comp[idx]*lat_std[idx]
        return delta