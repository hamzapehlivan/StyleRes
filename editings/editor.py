from .interfacegan import InterFaceGAN
from .ganspace import GanSpace
from .styleclip import StyleClip
from .gradctrl import GradCtrl
from options import Settings

"""
Entry class for all the edits. 
"""
class Editor():
    def __init__(self) -> None:
        self.interfacegan_editor = InterFaceGAN()
        self.ganspace_editor = GanSpace()
        self.styleclip_editor = StyleClip()
        self.gradctrl_editor = GradCtrl()

    def edit(self, latent, cfg):
        # Finds the corresponding function using method name
        if cfg.method == 'inversion':
            return latent

        editor = getattr(self, f'{cfg.method}_editor')
        return editor.edit(latent, cfg)

        