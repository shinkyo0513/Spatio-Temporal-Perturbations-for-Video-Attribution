import os
from os.path import join, isdir, isfile

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

from utils.PointingGame import PointingGame

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import numpy as np
from skimage import transform
from skimage.filters import gaussian

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ucf101", 
                                    choices=["ucf101", "epic"])
parser.add_argument("--model", type=str, default="r2p1d", 
                                    choices=["r2p1d", "v16l", "r50l"])   
parser.add_argument("--merge_masks", action='store_true')   
parser.add_argument("--vis_method", type=str, choices=["g", "ig", "sg", "sg2", "grad_cam", "perturb", "eb", "la", "gbp"])     
parser.add_argument('--only_test', action='store_true')
parser.add_argument('--only_train', action='store_true')         
parser.add_argument('--extra_label', type=str, default="")   
parser.add_argument('--smooth_sigma', type=int, default=0)    
# parser.add_argument("--res_buf", type=str)
# parser.add_argument("--tolerance", type=int, default=15, choices=[7, 15])
args = parser.parse_args()

phase_label = ""
if args.only_test:
    phase_label = "_test"
if args.only_train:
    phase_label = "_train"

if args.dataset == "ucf101":
    num_classes = 24
    ds_path = f'{ds_root}/UCF101_24'
    if args.model == "r2p1d" or args.model == "r50l":
        from datasets.ucf101_24_dataset_new import UCF101_24_Dataset as dataset
    elif args.model == "v16l":
        from datasets.ucf101_24_dataset_vgg16lstm import UCF101_24_Dataset as dataset
elif args.dataset == "epic":
    num_classes = 20
    ds_path = f'{ds_root}/epic'
    if args.model == "r2p1d" or args.model == "r50l":
        from datasets.epic_kitchens_dataset_new import EPIC_Kitchens_Dataset as dataset
    elif args.model == "v16l":
        from datasets.epic_kitchens_dataset_vgg16lstm import EPIC_Kitchens_Dataset as dataset

if args.model == "r2p1d" or args.model == "r50l":
    tolerance = 7
elif args.model == "v16l":
    tolerance = 15

# assert args.model in args.res_buf
# assert args.dataset in args.res_buf
# # # assert isfile(args.res_buf), f"Given directory of res_buf is invalid: {args.res_buf}"
res_buf = os.path.join(proj_root, 'exe_res', 
            f'{args.dataset}_{args.model}_{args.vis_method}_full{args.extra_label}{phase_label}.pt')
if args.vis_method == 'perturb':
    res_buf = res_buf.replace('.pt', '_summed.pt')

print(f'Loading videos ...')
video_dataset = dataset(ds_path, 16, 'long_range_last', 1, 6, False, bbox_gt=True, testlist_idx=1)
video_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=128)
video_dict = {}
for sample in tqdm(video_dataloader):
    clips_tensor = sample[0][0]
    cls_id = sample[1][0]
    video_name = sample[2][0].split("/")[-1]
    clip_bbox_tensor = sample[4][0]
    video_dict[video_name] = [clips_tensor, cls_id, clip_bbox_tensor]

pg_recorder = PointingGame(num_classes, tolerance)
# res_dic_lst = torch.load(args.res_buf)['val']
print('Loading', res_buf)
res_dic_lst = torch.load(res_buf)['val']
print(f'Loaded {res_buf}')
for res_dic in res_dic_lst:
    video_name = res_dic["video_name"]
    masks = res_dic["mask"]                 # numpy, (num_ares) x 1 x num_f x 14 x 14
    if args.merge_masks:
        masks = np.mean(masks, axis=0)
    masks = masks.squeeze(0)
    clips = video_dict[video_name][0].numpy()   # numpy, 3 x num_f x H x W
    cls_id = video_dict[video_name][1]          # int
    bboxes = video_dict[video_name][2].numpy()  # numpy, num_f x 4
    num_f = bboxes.shape[0]

    resize = (clips.shape[2:])

    for fidx in range(num_f):
        bbox = bboxes[fidx].tolist()
        mask = masks[fidx].astype(np.float32)  # numpy, 14x14
        resized_mask = transform.resize(mask, resize, order=1, mode='symmetric')  # numpy, H x W
        resized_mask = gaussian(resized_mask, sigma=args.smooth_sigma)
        if bbox != (0,0,0,0):
            max_ind = np.unravel_index(np.argmax(resized_mask, axis=None), resized_mask.shape)
            hit = pg_recorder.evaluate(bbox, max_ind)
            pg_recorder.aggregate(hit, cls_id)

print(pg_recorder)

