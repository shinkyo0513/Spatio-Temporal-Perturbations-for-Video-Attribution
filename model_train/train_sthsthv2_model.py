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
import json
import time

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
from utils.ReadingDataset import loadTags

crt_dir = os.path.dirname(os.path.realpath(__file__))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='r2p1d', choices=['r2p1d', 'r50l', 'tsm'])
parser.add_argument("--num_f", type=int, default=16, choices=[8, 16])
parser.add_argument("--long_range", action='store_true')
parser.add_argument("--num_epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=48)
parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--schedule_step", type=int, default=6)
parser.add_argument("--testlist_idx", type=int, default=1, choices=[1, 2])
parser.add_argument("--only_test", action='store_true')
parser.add_argument("--multi_gpu", action='store_true')
parser.add_argument("--retrain_type", type=str, default="Full")
parser.add_argument("--extra_label", type=str, default="")
parser.add_argument("--train_set", type=str, default="7000")
parser.add_argument("--labels_set", type=str, default="top25")
args = parser.parse_args()

if args.model == "r2p1d":
    from model_def.r2plus1d import r2plus1d as model
elif args.model == "r50l":
    from model_def.r50lstm import r50lstm as model
elif args.model == "tsm":
    from model_def.tsm import tsm as model
from datasets.sthsthv2_dataset_new import SthSthV2_Dataset

ds_name = "sthsthv2"
ds_path = os.path.join(ds_root, 'something_something_v2')

save_label = f"{ds_name}_{args.model}_{args.num_f}"
save_label = save_label + "_" + args.retrain_type
if args.long_range:
    save_label = save_label + "_LongRange"
if args.extra_label != "":
    save_label = save_label + f"_{args.extra_label}"

# Path to save and read model parameters
pt_save_dir = join(proj_root, 'model_param', f"{save_label}.pt")

multi_gpu = args.multi_gpu
num_devices = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.labels_set == 'top25':
    num_classes = 25
elif args.labels_set == 'full':
    num_classes = 174

if args.model == 'tsm':
    model_ft = model(num_classes=num_classes, segment_count=16, pretrained='sthsthv2')
    params_to_update = model_ft.parameters()
    model_ft.to_device(device)
else:
    model_ft = model(num_classes=num_classes, pretrained=True)
    model_ft.set_parameter_requires_grad(args.retrain_type)
    params_to_update = model_ft.get_parameter_to_update(debug=False)
    model_ft.to_device(device)
    if multi_gpu:
        model_ft.parallel_model(device_ids=list(range(num_devices)))

if not args.only_test:
    #================== Train & Evaluate ===================#
    optimizer_ft = torch.optim.SGD(params_to_update, lr=args.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, args.schedule_step, gamma=0.1)
    # criterion = nn.NLLLoss()
    if args.model == 'r2p1d':
        criterion = nn.CrossEntropyLoss()
    elif args.model == 'r50l':
        criterion = nn.CrossEntropyLoss()
    elif args.model == 'tsm':
        criterion = nn.CrossEntropyLoss()

    # Dataset and dataloader
    video_datasets = {}
    for x in ['train', 'val']:
        if args.long_range:
            sample_mode = 'long_range_random' if x == 'train' else 'long_range_last'
            num_clips = 1 if x == 'train' else 1
            frame_rate = 2
        else:
            sample_mode = "random" if x=="train" else "fixed"
            num_clips = 1
            frame_rate = 2
        video_datasets[x] = SthSthV2_Dataset(ds_path, args.num_f, sample_mode, num_clips, 
                                            frame_rate, x=='train', testlist_idx=args.testlist_idx, 
                                            labels_set=args.labels_set, train_set=args.train_set)
    print({x: 'Num of clips:{}'.format(len(video_datasets[x])) for x in ['train', 'val']})
    dataloaders = {x: DataLoader(video_datasets[x], batch_size=args.batch_size, shuffle=True, 
                        num_workers=128) for x in ['train', 'val']}
    # Train
    hist = model_ft.train_model(dataloaders, criterion, optimizer_ft, pt_save_dir, args.num_epochs, scheduler)

#========================== Only validate =========================#
sample_mode = 'long_range_last' if args.long_range else "fixed"
val_dataset = SthSthV2_Dataset(ds_path, args.num_f, sample_mode, 1, 2, train=False, 
                               testlist_idx=args.testlist_idx, labels_set=args.labels_set)
print('Num of clips:{}'.format(len(val_dataset)))
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, 
                    num_workers=128)
# Validate
if args.model == 'tsm':
    y_pred, y_true = model_ft.val_model(val_dataloader)
else:
    y_pred, y_true = model_ft.val_model(val_dataloader, pt_save_dir)

# # # Draw confusion matrix
# annot_path = os.path.join(proj_root, 'my_sthsthv2_annot', 'top25_labels_index.json')
# with open(annot_path) as f:
#     labels_index = json.load(f)
# class_names = list(labels_index.keys())
# cm_fig, _ = plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)
# cm_fig.savefig(os.path.join(proj_root, 'visual_res', 'cm', f'{save_label}.png'))
