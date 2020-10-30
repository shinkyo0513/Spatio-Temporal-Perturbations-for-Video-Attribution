#%%
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import os
from os.path import join

import time
import copy
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
from skimage.filters import gaussian

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

from utils.ConfusionMatrix import plot_confusion_matrix
from utils.CalAcc import AverageMeter, accuracy
from utils import ImageShow

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="r2p1d", choices=["r2p1d", "r50l"])
parser.add_argument("--num_f", type=int, default=16, choices=[8, 16])
parser.add_argument("--testlist_idx", type=int, default=1, choices=[1, 2])
parser.add_argument("--vis_method", type=str, choices=["g", "ig", "sg", "sg2", "grad_cam", "perturb", "random"])
parser.add_argument("--mode", type=str, default="ins", choices=["ins", "del"])
parser.add_argument("--specify_video_name", type=str, default="")
parser.add_argument("--perturb_ratio", type=float)
parser.add_argument("--smoothed_perturb", action='store_true')
parser.add_argument("--smooth_sigma", type=int, default=10, choices=[5, 10])
parser.add_argument("--perturb_by_block", action='store_true')
parser.add_argument("--noised", action='store_true')
args = parser.parse_args()

from datasets.ucf101_24_perturb_dataset_new import UCF101_24_Dataset
num_classes = 24
ds_name = "ucf101_24"
ds_path = f"{ds_root}/UCF101_24/"

save_label = f"{ds_name}_{args.model}_{args.vis_method}_{args.perturb_ratio}"
if args.smoothed_perturb:
    save_label = save_label + '_smoothed' + f'{args.smooth_sigma}'
if args.perturb_by_block:
    save_label = save_label + '_block'
if args.noised:
    save_label = save_label + '_noised'
save_path = join(proj_root, 'visual_res', f"{save_label}")
os.makedirs(save_path, exist_ok=True)

num_devices = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

heatmaps_dir = os.path.join(proj_root, 'exe_res', f'ucf101_{args.model}_{args.vis_method}_full.pt')
if args.vis_method == 'perturb':
    heatmaps_dir = heatmaps_dir.replace('.pt', '_summed.pt')
elif args.vis_method == 'random':
    heatmaps_dir = 'random'

#========================== Only validate =========================#
sample_mode = 'long_range_last'
frame_rate = 6
val_dataset = UCF101_24_Dataset(ds_path, args.num_f, sample_mode, 1, 
                                heatmaps_dir, args.perturb_ratio,
                                frame_rate, train=False, 
                                testlist_idx=args.testlist_idx,
                                smoothed_perturb=args.smoothed_perturb,
                                smooth_sigma=args.smooth_sigma,
                                perturb_by_block=args.perturb_by_block,
                                noised=args.noised)
print('Num of clips:{}'.format(len(val_dataset)))
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=128)

for samples in tqdm(val_dataloader):
    perturbed_clip_tensor = samples[0][0]
    video_name = samples[2][0]
    heatmaps_tensor = samples[4][0]

    video_name_regu = video_name.split('/')[-1]

    ImageShow.plot_voxel(perturbed_clip_tensor, heatmaps_tensor, 
                save_path=os.path.join(save_path, video_name_regu+'.jpg'))
