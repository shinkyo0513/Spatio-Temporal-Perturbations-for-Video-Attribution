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
from tqdm import tqdm
import math
import numpy as np
from skimage import transform

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cat_ucf", choices=["cat_ucf"])
parser.add_argument("--model", type=str, default='r2p1d', choices=['r2p1d', 'r50l', 'v16l'])
parser.add_argument("--vis_method", type=str, choices=["g", "ig", "sg", "sg2", "grad_cam", "perturb"])    
parser.add_argument('--extra_label', type=str, default="")
# parser.add_argument("--tolerance", type=int, default=15, choices=[7, 15])
args = parser.parse_args()

num_classes = 24
ds_path = f'{ds_root}/Cat_UCF101'
if args.model == "r2p1d":
    from datasets.cat_ucf_testset_new import Cat_UCF_Testset as dataset
    # from model_def.r2plus1d import r2plus1d as model
    # model_wgts_dir = f"{proj_root}/model_param/ucf101_24_r2p1d_16_Full_LongRange.pt"
elif args.model == "r50l":
    from datasets.cat_ucf_testset_new import Cat_UCF_Testset as dataset
    # from model_def.r50lstm import r50lstm as model
    # model_wgts_dir = f"{proj_root}/model_param/ucf101_24_r50l_16_Full_LongRange.pt"

res_buf = os.path.join(proj_root, 'exe_res', 
            f'{args.dataset}_{args.model}_{args.vis_method}_full{args.extra_label}.pt')
if args.vis_method == 'perturb':
    res_buf = res_buf.replace('.pt', '_summed.pt')

video_pair_file = f'{ds_root}/Cat_UCF101/UCF101_Cat_Pair.txt'
ground_mask_dict = {}
with open(video_pair_file, 'r') as f:
    for line in f.readlines():
        video_name, paired_video_name, gt_side, gt_numf = line.strip().split(' ')
        gt_numf = int(gt_numf)
        video_name = video_name.split('/')[-1]
        ground_mask = [1,]*gt_numf+[0,]*(16-gt_numf)  if gt_side=='left' else [0,]*(16-gt_numf)+[1,]*gt_numf
        ground_mask = torch.tensor(ground_mask).to(torch.long)
        ground_mask_dict[video_name] = ground_mask

# cls_wise_hits = np.zeros((num_classes))
hits = 0
res_dic_lst = torch.load(res_buf)['val']
for res_dic in res_dic_lst:
    video_name = res_dic["video_name"]
    masks = res_dic["mask"]                     # numpy, num_area x num_f x 14 x 14
    ground_mask = ground_mask_dict[video_name]

    tsal = masks.reshape((16, -1)).sum(axis=1)   # num_f
    hlt = np.argmax(tsal)
    if ground_mask[hlt] == 1:
        # cls_wise_hits[cls_id] += 1
        hits += 1
# hit_rate = cls_wise_hits.sum() / len(res_dic_lst)
hit_rate = hits / len(res_dic_lst)
print(f"Ave hit rate: {hit_rate:.2f}")
# print(f"Class-wise hit time: {cls_wise_hits.tolist()}")


