from argparse import ArgumentParser
import os
import sys
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from evaluation.recons_metrics.criteria.lpips.lpips import LPIPS
from evaluation.recons_metrics.criteria.ms_ssim import MSSSIM
from evaluation.recons_metrics.gt_res_dataset import GTResDataset


def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--mode', type=str, default='lpips', choices=['lpips', 'l2', 'msssim'])
    parser.add_argument('--data_path', type=str, default='results')
    parser.add_argument('--gt_path', type=str, default='gt_images')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--is_cars', action='store_true')
    args = parser.parse_args()
    return args


def run(args):
    resize_dims = (256, 256)
    if args.is_cars:
        resize_dims = (192, 256)
    transform = transforms.Compose([transforms.Resize(resize_dims),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    print('Loading dataset')
    dataset = GTResDataset(root_path=args.data_path,
                           gt_dir=args.gt_path,
                           transform=transform)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=int(args.workers),
                            drop_last=False)

    if args.mode == 'lpips':
        loss_func = LPIPS(net_type='alex')
    elif args.mode == 'l2':
        loss_func = torch.nn.MSELoss()
    elif args.mode == 'msssim':
        loss_func = MSSSIM()
    else:
        raise Exception('Not a valid mode!')
    loss_func.cuda()

    global_i = 0
    scores_dict = {}
    all_scores = []
    for result_batch, gt_batch in tqdm(dataloader):
        for i in range(args.batch_size):
            loss = float(loss_func(result_batch[i:i + 1].cuda(), gt_batch[i:i + 1].cuda()))
            all_scores.append(loss)
            im_path = dataset.pairs[global_i][0]
            scores_dict[os.path.basename(im_path)] = loss
            global_i += 1

    all_scores = list(scores_dict.values())
    mean = np.nanmean(all_scores)
    std = np.nanstd(all_scores)
    result_str = '{}: {}+-{:.2f}'.format(args.mode,mean, std)
    print('Finished with ', args.data_path)
    print(result_str)


if __name__ == '__main__':
    args = parse_args()
    run(args)
