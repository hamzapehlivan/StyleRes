import torch
import os
import numpy as np
import pickle
from editings.gradctrl_manipulator.classifiers import *
from options import Settings

EXCLUDE_DICT_FFHQ = {0:[[1,2,3],[250,100,100],[2,10]],1:[[0,2,3],[200,50,100], [2,8]],
        2:[[0,1,3],[200,100,250],[1,4]], 3:[[0,2],[100,300],[4,10]]}


class GradManipulator():
    def __init__(self) -> None:
        self.model_dir = Settings.gradctrl_modeldir
        self.is_clf_initialized = False
        
    def init_clf(self):
        if not self.is_clf_initialized:
            with open(os.path.join(self.model_dir , 'model_params.pkl'), 'rb') as handle:
                model_params = pickle.load(handle)
            self.clf = MultiLabelClassifier(**model_params)
            save_path = os.path.join(self.model_dir , 'model.pth')
            self.clf.load_state_dict(torch.load(save_path, map_location='cpu'), strict=False)
            self.clf.to(Settings.device)
            self.is_clf_initialized = True

    def grad_cam(self, C, c_to_optimize, latents):

        if not latents.requires_grad:
            latents.requires_grad=True
        #mini batch size=1 to avoid oom
        fz_dzs = []
        
        for i in range(latents.shape[0]):
            latent = latents[i].unsqueeze(0)
            predictions, _ = C(latent)
            predictions = predictions.squeeze()[c_to_optimize]
            fz_dz = torch.autograd.grad(outputs=predictions,
                                    inputs= latent,
                                    retain_graph=True,
                                    create_graph= True,
                                    allow_unused=True
                                    )[0].detach()
            fz_dzs.append(fz_dz)  
        grad = torch.cat(fz_dzs, dim=0)

        grad_cam = torch.abs(grad)
        return grad_cam, grad

    def optimize_latent(self, w, num_steps, learning_rate, edit, direction, **unused_kwargs):

        direction = 1 if direction == 'negative' else -1
        c_to_optimize = self.clf.attributes.index(edit)
    
        editing_params = EXCLUDE_DICT_FFHQ[c_to_optimize]
        c_to_excludes = editing_params[0]   #exclude
        top_cs = editing_params[1]          #top_channels
        [layer_lower,layer_upper] = editing_params[2]

        edit_directions = []
        for l_i, l in enumerate(w):
            orig_latent = l.clone().detach().unsqueeze(0)
           
            edited_latent = orig_latent.clone().to(Settings.device)
            edited_latent.requires_grad = True
            for s in range(num_steps):
                dims_to_exclude = []
                predictions_full = self.clf(edited_latent)
                predictions = predictions_full[0].squeeze()
                #compute editing direction
                fz_dz_target = torch.autograd.grad(outputs=predictions[c_to_optimize],
                                                        inputs= edited_latent,
                                                        retain_graph=True,
                                                        create_graph= True,
                                                        allow_unused=True
                                                        )[0]

                #disentanglement through channel filtering                                        
                for c_idx, c_to_exclude in enumerate(c_to_excludes):
                    dim_c,_ = self.grad_cam(self.clf, c_to_exclude, edited_latent)
                    dim_c = dim_c.detach().squeeze().cpu().numpy()
                    excluded = np.argsort(dim_c)[-top_cs[c_idx]:]
                    dims_to_exclude.append(excluded)
                if len(dims_to_exclude)>0:
                    dims_to_exclude = np.unique(np.concatenate(dims_to_exclude))
                    mask = torch.ones(512).to(Settings.device)
                    mask[dims_to_exclude] = 0
                    fz_dz_target *= mask
                fz_dz_target /= torch.norm(fz_dz_target)

                with torch.no_grad():
                    edited_latent += fz_dz_target*learning_rate*direction

            with torch.no_grad():
                edit_directions.append(edited_latent)

        with torch.no_grad():
            edit_directions = torch.cat( edit_directions, dim=0)
            w_plus = torch.repeat_interleave(w.unsqueeze(1), 18, dim=1)
            w_plus[:, layer_lower:layer_upper,:] = edit_directions.unsqueeze(1)
            edit_direction = w.unsqueeze(1) - w_plus
        return edit_direction