import torch
from options import Settings
import os

class InterFaceGAN():
    def __init__(self) -> None:
        pass

    def edit(self, latent, cfg):
        with torch.no_grad():
            return latent + cfg.strength * self.get_direction(cfg.edit)

    def get_direction(self, editname):
        try:
            direction = getattr(self, f"{editname}_direction")
        except:
            direction = self.load_direction(editname)
            if Settings.device != 'cpu':
                direction = direction.to(Settings.device)
            setattr(self, f"{editname}_direction", direction.clone())
        return direction

    def load_direction(self, editname):
        direction = torch.load(os.path.join( Settings.interfacegan_directions, f'{editname}.pt'))
        if Settings.device != 'cpu':
            direction = direction.cuda()
        return direction