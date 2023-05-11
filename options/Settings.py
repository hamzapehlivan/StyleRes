"""
Global settings file for all the classes. Only inference.py should write here.
Other files can read from this file.
"""

device = 'cpu'
interfacegan_directions = 'editings/interfacegan_directions'
ganspace_directions = 'editings/ganspace_pca'
styleclip_settings = 'editings/styleclip_directions'
styleclip_mapper_directions = 'editings/styleclip_directions/styleclip_directions/mapper'
styleclip_global_directions = 'editings/styleclip_directions/styleclip_directions/global_directions'
s_statistics = 'editings/styleclip_directions/styleclip_directions/global_directions/ffhq/S_mean_std'
text_prompt_templates = 'editings/styleclip_directions/styleclip_directions/global_directions/templates.txt'
delta_i_c = 'editings/styleclip_directions/styleclip_directions/global_directions/ffhq/fs3.npy'
gradctrl_modeldir = 'editings/gradctrl_manipulator/model_ffhq'