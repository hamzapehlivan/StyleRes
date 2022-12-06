import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.inference_dataset import InferenceDataset
from datasets.process_image import ImageProcessor
from models.styleres import StyleRes
from options import Settings
from options.inference_options import InferenceOptions
from utils import parse_config


def initialize_styleres(checkpoint_path, device):
    Settings.device = device
    model = StyleRes()
    model.load_ckpt(checkpoint_path)
    model.send_to_device()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def run():
    args = InferenceOptions().parse()
    edit_configs = parse_config(args.edit_configs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = InferenceDataset(args.datadir, aligner_path=args.aligner_path)
    print(f"Dataset is created. Number of images is {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size = args.test_batch_size,
                            shuffle=False,
                            num_workers=int(args.test_workers),
                            drop_last=False)

    if args.n_images == None:
        args.n_images = len(dataset)

    # Create output directories
    output_dir = args.outdir
    os.makedirs(output_dir, exist_ok=True)
    for edit_config in edit_configs:
        cfg_vals = edit_config.values()
        edit_config.outdir = '_'.join( str(i) for i in cfg_vals)
        os.makedirs( os.path.join(output_dir, edit_config.outdir), exist_ok=True)

    resize_amount = (1024, 1024)
    if args.resize_outputs:
        resize_amount = (256,256)
    
    # Setup model
    model = initialize_styleres(args.checkpoint_path, device)
        
    n_images = 0
    for data in tqdm(dataloader):
        if n_images >= args.n_images:
            break
        n_images = n_images + data['image'].shape[0]
        for edit_config in edit_configs:
            images = model.edit_images( data['image'], edit_config)
            images = ImageProcessor.postprocess_image(images.detach().cpu().numpy())
            for j in range( images.shape[0]):
                save_name = data['name'][j]
                pil_img = Image.fromarray(images[j]).resize(resize_amount)
                pil_img.save(os.path.join(output_dir,  edit_config.outdir, save_name))


if __name__ == '__main__':
    run()
