import torch
try:
    import clip
except:
    print("CLIP model is not available. Global directions method can't be used")
import copy
from options import Settings

"""
Modified from HyperStyle repository
https://github.com/yuval-alaluf/hyperstyle/blob/main/editing/styleclip/global_direction.py
"""
STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]

TORGB_INDICES = list(range(1, len(STYLESPACE_DIMENSIONS), 3))
STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in TORGB_INDICES][:11]

def features_channels_to_s(s_without_torgb, s_std):
    s = []
    start_index_features = 0
    for c in range(len(STYLESPACE_DIMENSIONS)):
        if c in STYLESPACE_INDICES_WITHOUT_TORGB:
            end_index_features = start_index_features + STYLESPACE_DIMENSIONS[c]
            s_i = s_without_torgb[start_index_features:end_index_features] * s_std[c]
            start_index_features = end_index_features
        else:
            s_i = torch.zeros(STYLESPACE_DIMENSIONS[c]).to(Settings.device)
        s_i = s_i.view(1, 1, -1, 1, 1)
        s.append(s_i)
    return s

class StyleCLIPGlobalDirection:

    def __init__(self, delta_i_c, s_std, text_prompts_templates):
        super(StyleCLIPGlobalDirection, self).__init__()
        self.delta_i_c = delta_i_c
        self.s_std = s_std
        self.text_prompts_templates = text_prompts_templates
        self.clip_model, _ = clip.load("ViT-B/32", device=Settings.device)

    def get_delta_s(self, neutral_text, target_text, beta):
        delta_i = self.get_delta_i([target_text, neutral_text]).float()
        r_c = torch.matmul(self.delta_i_c, delta_i)
        delta_s = copy.copy(r_c)
        channels_to_zero = torch.abs(r_c) < beta
        delta_s[channels_to_zero] = 0
        max_channel_value = torch.abs(delta_s).max()
        if max_channel_value > 0:
            delta_s /= max_channel_value
        direction = features_channels_to_s(delta_s, self.s_std)
        return direction

    def get_delta_i(self, text_prompts):
        try:   # Check if loaded
            delta_i = getattr(self, f"{text_prompts[0]}_{text_prompts[1]}")
        except:
            text_features = self._get_averaged_text_features(text_prompts)
            delta_t = text_features[0] - text_features[1]
            delta_i = delta_t / torch.norm(delta_t)
            setattr(self, f"{text_prompts[0]}_{text_prompts[1]}", delta_i)
        return delta_i

    def _get_averaged_text_features(self, text_prompts):
        with torch.no_grad():
            text_features_list = []
            for text_prompt in text_prompts:
                formatted_text_prompts = [template.format(text_prompt) for template in self.text_prompts_templates]  # format with class
                formatted_text_prompts = clip.tokenize(formatted_text_prompts).to(Settings.device)  # tokenize
                text_embeddings = self.clip_model.encode_text(formatted_text_prompts)  # embed with text encoder
                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                text_embedding = text_embeddings.mean(dim=0)
                text_embedding /= text_embedding.norm()
                text_features_list.append(text_embedding)
            text_features = torch.stack(text_features_list, dim=1).to(Settings.device)
        return text_features.t()