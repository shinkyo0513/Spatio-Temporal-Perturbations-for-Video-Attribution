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
import csv

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
from ReadingDataset import loadTags

crt_dir = os.path.dirname(os.path.realpath(__file__))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='r2p1d', choices=['r2p1d', 'v16l', 'v16l_small', 'r50l'])
parser.add_argument("--num_f", type=int, default=16, choices=[8, 16])
parser.add_argument("--long_range", action='store_true')
parser.add_argument("--num_epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=48)
parser.add_argument("--testlist_idx", type=int, default=2, choices=[1, 2])
parser.add_argument("--only_test", action='store_true')
parser.add_argument("--multi_gpu", action='store_true')
parser.add_argument("--retrain_type", type=str, default="Full")
parser.add_argument("--vis_method", type=str, choices=["g", "ig", "sg", "sg2", "grad_cam", "perturb", "random"])
parser.add_argument("--perturb_ratio", type=float)
parser.add_argument("--smoothed_perturb", action='store_true')
parser.add_argument("--smooth_sigma", type=int, default=10, choices=[5, 10])
parser.add_argument("--perturb_by_block", action='store_true')
parser.add_argument("--noised", action='store_true')
args = parser.parse_args()

if args.model == "r2p1d":
    from model_def.r2plus1d import r2plus1d as model
    from datasets.epic_kitchens_perturb_dataset_new import EPIC_Kitchens_Dataset
elif args.model == "r50l":
    from model_def.r50lstm import r50lstm as model
    from datasets.epic_kitchens_perturb_dataset_new import EPIC_Kitchens_Dataset
elif args.model == "v16l":
    from model_def.vgg16lstm import vgg16lstm as model
    from datasets.epic_kitchens_perturb_dataset_vgg16lstm import EPIC_Kitchens_Dataset

num_classes = 20
ds_name = "epic"
ds_path = f"{ds_root}/epic/"

save_label = f"{ds_name}_{args.model}_{args.num_f}"
save_label = save_label + "_" + args.retrain_type
save_label = save_label + f"_{args.vis_method}_{args.perturb_ratio}"
if args.smoothed_perturb:
    save_label = save_label + '_smoothed' + f'{args.smooth_sigma}'
if args.perturb_by_block:
    save_label = save_label + '_block'
if args.long_range:
    save_label = save_label + "_LongRange"

multi_gpu = args.multi_gpu
num_devices = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ft = model(num_classes=num_classes, pretrained=True)
model_ft.set_parameter_requires_grad(args.retrain_type)
params_to_update = model_ft.get_parameter_to_update(debug=False)
model_ft.to_device(device)
if multi_gpu:
    model_ft.parallel_model(device_ids=list(range(num_devices)))

heatmaps_dir = os.path.join(proj_root, 'exe_res', f'ucf101_{args.model}_{args.vis_method}_full.pt')
if args.vis_method == 'perturb':
    heatmaps_dir = heatmaps_dir.replace('.pt', '_summed.pt')
elif args.vis_method == 'random':
    heatmaps_dir = 'random'

# Path to save and read model parameters
pt_save_dir = join(proj_root, 'model_param', f"{save_label}.pt")

if not args.only_test:
    #================== Train & Evaluate (for EPIC) ===================#
    # OPtimizer and Loss fn
    if args.model == "r50l":
        optimizer_ft = torch.optim.SGD(params_to_update, lr=0.05, momentum=0.9)
    else:
        optimizer_ft = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    # criterion = nn.NLLLoss()
    if args.model == "r2p1d" or args.model == "r50l":
        criterion = nn.CrossEntropyLoss()
    elif args.model == "v16l":
        criterion = nn.NLLLoss()

    # Dataset and dataloader
    video_datasets = {}
    for x in ['train', 'val']:
        if args.long_range:
            sample_mode = 'long_range_last'
            num_clips = 1
            frame_rate = 6
        else:
            sample_mode = "fixed"
            num_clips = 1
            frame_rate = 6
        video_datasets[x] = EPIC_Kitchens_Dataset(ds_path, args.num_f, sample_mode, num_clips, 
                                                    heatmaps_dir, args.perturb_ratio,
                                                    frame_rate, x=='train', 
                                                    testlist_idx=args.testlist_idx
                                                    smoothed_perturb=args.smoothed_perturb,
                                                    smooth_sigma=args.smooth_sigma,
                                                    perturb_by_block=args.perturb_by_block,
                                                    noised=args.noised)
    print({x: 'Num of clips:{}'.format(len(video_datasets[x])) for x in ['train', 'val']})
    dataloaders = {x: DataLoader(video_datasets[x], batch_size=args.batch_size, shuffle=True, 
                        num_workers=128) for x in ['train', 'val']}

    if model == 'r50l':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, 15, gamma=0.1)
    elif model == 'r2p1d':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, 6, gamma=0.1)
    else:
        scheduler = None

    # Train
    hist = model_ft.train_model(dataloaders, criterion, optimizer_ft, pt_save_dir, args.num_epochs, scheduler)

#========================== Only validate =========================#
sample_mode = 'long_range_last' if args.long_range else "fixed"
val_dataset = EPIC_Kitchens_Dataset(ds_path, args.num_f, sample_mode, 1, 
                                    heatmaps_dir, args.perturb_ratio,
                                    6, train=False, 
                                    testlist_idx=args.testlist_idx,
                                    smoothed_perturb=args.smoothed_perturb,
                                    smooth_sigma=args.smooth_sigma,
                                    perturb_by_block=args.perturb_by_block,
                                    noised=args.noised)
print('Num of clips:{}'.format(len(val_dataset)))
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, 
                    num_workers=128)
# Validate
y_pred, y_true = model_ft.val_model(val_dataloader, pt_save_dir)

# # Draw confusion matrix
tags,tag2ID = loadTags(join(proj_root, 'datasets', 'epic_top20_catName.txt'))
class_names = np.asarray(tags)
cm_fig, _ = plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)
cm_fig.savefig(os.path.join(proj_root, 'visual_res', 'cm', f'{save_label}.png'))
