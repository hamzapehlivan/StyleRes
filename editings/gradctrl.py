from .gradctrl_manipulator.manipulate import GradManipulator

class GradCtrl:
    def __init__(self) -> None:
        self.grad_manipulator = GradManipulator()

    def edit(self, latent, cfg):
        self.grad_manipulator.init_clf()
        edit_dir = self.grad_manipulator.optimize_latent(latent[:,0,:], **cfg)
        return latent + edit_dir
        
        
