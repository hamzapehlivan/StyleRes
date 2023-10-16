attribute_to_idx = {
    '5_o_Clock_Shadow': 1,
    'Arched_Eyebrows': 2,
    'Attractive': 3,
    'Bags_Under_Eyes': 4,
    'Bald': 5,
    'Bangs': 6,
    'Big_Lips': 7,
    'Big_Nose': 8,
    'Black_Hair': 9,
    'Blond_Hair': 10,
    'Blurry': 11,
    'Brown_Hair': 12,
    'Bushy_Eyebrows': 13,
    'Chubby': 14,
    'Double_Chin': 15,
    'Eyeglasses': 16,
    'Goatee': 17,
    'Gray_Hair': 18,
    'Heavy_Makeup': 19,
    'High_Cheekbones': 20,
    'Male': 21,
    'Mouth_Slightly_Open': 22,
    'Mustache': 23,
    'Narrow_Eyes': 24,
    'No_Beard': 25,
    'Oval_Face': 26,
    'Pale_Skin': 27, 
    'Pointy_Nose': 28,
    'Receding_Hairline': 29,
    'Rosy_Cheeks': 30,
    'Sideburns': 31, 
    'Smiling': 32,
    'Straight_Hair': 33,
    'Wavy_Hair': 34,
    'Wearing_Earrings': 35,
    'Wearing_Hat': 36,
    'Wearing_Lipstick': 37,
    'Wearing_Necklace': 38,
    'Wearing_Necktie': 39,
    'Young': 40
}
set_to_idx = {
    'train': 0,
    'eval': 1,
    'val': 1,
    'validation': 1,
    'test': 2
}

import argparse
from PIL import Image
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='data/CelebAMask-HQ/CelebA-HQ-img')
parser.add_argument('--annotations', type=str, default='data/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt')
parser.add_argument('--partitions', type=str, default='data/CelebAMask-HQ/list_eval_partition.txt')
parser.add_argument('--mapping', type=str, default='data/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt')
parser.add_argument('--attribute', nargs='+', type=str, default=None, help='Which attribute to extract followed by label. Ex: Smiling -1')
parser.add_argument('--set', type=str, default='test', help="From which dataset to extract")
parser.add_argument('--output', type=str, default='data/celebahq_test')
args = parser.parse_args()


anns = np.loadtxt(args.annotations, skiprows=2, dtype=str)
mapping = np.loadtxt(args.mapping, skiprows=1, usecols=1, dtype=int)
partitions = np.loadtxt(args.partitions, usecols=1, dtype=int)

dataset_idx = set_to_idx[args.set]
indices_in_set = partitions[ mapping.tolist()] == dataset_idx

if (args.attribute != None): 
    attr_idx = attribute_to_idx[args.attribute[0]]
    indices_with_attr = anns[:, attr_idx] == args.attribute[1]
    indices_in_set = indices_in_set & indices_with_attr

file_names = anns[ indices_in_set][:,0]
os.makedirs(args.output)
for name in file_names:
    image = Image.open( os.path.join(args.dataset_path, name))
    image = image.resize((256,256))
    image.save( os.path.join(args.output, name))
