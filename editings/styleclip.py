
from editings.styleclip_directions.styleclip_mapper_network import LevelsMapper
from editings.styleclip_directions.styleclip_global_calculator import StyleCLIPGlobalDirection
import torch
import csv
from options import Settings
import os
import numpy as np
import pickle

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
                calculator = self.build_global_direction_calculator()
                directions = calculator.get_delta_s(cfg.neutral_text, cfg.target_text, cfg.disentanglement)
                edited_styles = []
                for style, direction in zip(latent, directions):
                    direction = direction.view(-1, style.shape[1])
                    edited_styles.append( style + cfg.strength* direction )
                return edited_styles
                
    def build_global_direction_calculator(self):
        try:   # Check if loaded
            global_direction_calculator = getattr(self, f"global_direction_calculator")
        except:
            delta_i_c = torch.from_numpy(np.load(Settings.delta_i_c)).float().to(Settings.device)
            with open(Settings.s_statistics, "rb") as channels_statistics:
                _, s_std = pickle.load(channels_statistics)
                s_std = [torch.from_numpy(s_i).float().to(Settings.device) for s_i in s_std]
            with open(Settings.text_prompt_templates, "r") as templates:
                text_prompt_templates = templates.readlines()
            global_direction_calculator = StyleCLIPGlobalDirection(delta_i_c, s_std, text_prompt_templates)
            setattr(self, f"global_direction_calculator", global_direction_calculator)
        return global_direction_calculator

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