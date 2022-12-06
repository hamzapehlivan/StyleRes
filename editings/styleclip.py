
from editings.styleclip_directions.styleclip_mapper_network import LevelsMapper
import torch
import csv
from options import Settings
import os

class Options():
    def __init__(self, no_coarse_mapper, no_medium_mapper, no_fine_mapper) -> None:
        self.no_coarse_mapper = no_coarse_mapper
        self.no_medium_mapper = no_medium_mapper
        self.no_fine_mapper = no_fine_mapper

class StyleClip():
    def __init__(self) -> None:
        self.styleclip_mapping_configs = {}
        
        with open(os.path.join(Settings.styleclip_settings, 'styleclip_mapping_configs.csv'), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                key = row.pop(0)
                self.styleclip_mapping_configs[key] = list(map(lambda x: True if x == "True" else False, row))
        
    def edit(self, latent, cfg):
        with torch.no_grad():
            if cfg.type == 'mapper':
                mapper = self.build_mapper(cfg.edit)
                return latent + cfg.strength * mapper(latent)
            if cfg.type == 'global':
                
                return latent + 10 * torch.load(os.path.join(Settings.styleclip_global_directions, 'makeup.pt'))

    # def load_global_direction(self, editname):
    #     pass

    def build_mapper(self, editname):
        try:   # Check if loaded
            mapper = getattr(self, f"{editname}_mapper")
        except:
            opts = Options(*self.styleclip_mapping_configs[editname])
            mapper = LevelsMapper(opts)
            ckpt = torch.load(os.path.join(Settings.styleclip_mapper_directions, f'{editname}.pt'))
            mapper.load_state_dict(ckpt, strict=True)
            mapper.to(device=Settings.device)
            for param in mapper.parameters():
                param.requires_grad = False
            mapper.eval()
            setattr(self, f"{editname}_mapper", mapper)
        return mapper