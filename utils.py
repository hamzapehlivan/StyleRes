import sys
import os
from importlib import import_module
from options import Settings
import csv 

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


"""
    This function modified from the Genforce library: https://github.com/genforce/genforce
"""
def parse_config(config_file):
    """Parses configuration from python file."""
    assert os.path.isfile(config_file)
    directory = os.path.dirname(config_file)
    filename = os.path.basename(config_file)
    module_name, extension = os.path.splitext(filename)
    assert extension == '.py'
    sys.path.insert(0, directory)
    module = import_module(module_name)
    sys.path.pop(0)
    config = []
    for key, value in module.__dict__.items():
        if key.startswith('__'):
            continue
        for val in value:
            attr_dict = AttrDict()
            for k, v in val.items():
                attr_dict[k] = v
            config.append(attr_dict)
    del sys.modules[module_name]
    return config

# Utility class for the demo
class AppUtils():
    def __init__(self):
        self.interfacegan_edits = ['Smile', 'Age' , 'Pose']
        self.ganspace_edits = []
        with open(os.path.join(Settings.ganspace_directions, 'ganspace_configs.csv'), "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                key = row.pop(0)
                key = key.replace('_', ' ')
                self.ganspace_edits.append(key.title())
        self.ganspace_edits.sort()

        self.styleclip_edits = []
        with open(os.path.join(Settings.styleclip_settings, 'styleclip_mapping_configs.csv'), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                key = row.pop(0)
                key = key.replace('_', ' ')
                self.styleclip_edits.append(key.title())
        self.styleclip_edits.sort()

    def get_methods(self):
        return ["InterfaceGAN", "GANSpace", "StyleClip"]

    def get_edits(self, method):
        method = method.lower()
        return getattr(self, f"{method}_edits")

    def args_to_cfg(self, method, edit, strength):
        method = method.lower()
        edit = edit.lower()
        edit = edit.replace(' ', '_')
        strength = float(strength)
        cfg = AttrDict()
        cfg.method = method
        cfg.edit = edit
        cfg.strength = strength
        if method == 'styleclip':
            cfg.type = 'mapper'
        return cfg

    def get_range(self, method):
        method = method.lower()
        if method == 'interfacegan':
            return -5, 5, 0.1
        elif method == 'ganspace':
            return -25, 25, 0.1
        elif method == 'styleclip':
            return 0, 0.2, 0.01

    def get_examples(self):
        examples = [
            ["samples/demo_samples/11654.jpg", "InterfaceGAN",  "Age", 2.0,  False],
            ["samples/demo_samples/116.jpg", "Ganspace",  "lipstick", 10.0,  False],
            ["samples/demo_samples/carlsen.jpg", "Styleclip", "curly hair", 0.11, True],
            ["samples/demo_samples/shakira.jpeg", "StyleClip",  "purple hair",  0.1,  True],
            ["samples/demo_samples/shaq.jpg", "InterfaceGAN",  "Smile", -1.7,  True],
            ["samples/demo_samples/shaq.jpg", "InterfaceGAN",  "Pose", 3.3,   True]
        ]
        return examples

