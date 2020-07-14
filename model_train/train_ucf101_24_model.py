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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="vgg16lstm", choices=["r2plus1d_18", "vgg16lstm", "vgg16lstm_small"])
parser.add_argument("--num_f", type=int, default=16, choices=[8, 16])
parser.add_argument("--long_range", action='store_true')
parser.add_argument("--num_epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=48)
parser.add_argument("--testlist_idx", type=int, choices=[1, 2])
parser.add_argument("--only_test", action='store_true')
parser.add_argument("--retrain_type", type=str, default="FC+FinalConv",
                    choices=["FC+FinalConv", "Full"])
args = parser.parse_args()

if args.model == "r2plus1d_18":
    from model_def.r2plus1d import r2plus1d as model
    from datasets.ucf101_24_dataset_new import UCF101_24_Dataset
    # retrain_type = "FC+FinalConv"
    retrain_type = args.retrain_type
    num_classes = 24
elif args.model == "vgg16lstm":
    from model_def.vgg16lstm import vgg16lstm as model
    from datasets.ucf101_24_dataset_vgg16lstm import UCF101_24_Dataset
    num_classes = 24
    # retrain_type = "AllFC+LSTM"
    retrain_type = args.retrain_type
elif args.model == "vgg16lstm_small":
    from model_def.vgg16lstm_small import vgg16lstm as model
    from datasets.ucf101_24_dataset_vgg16lstm_small import UCF101_24_Dataset
    num_classes = 24
    # retrain_type = "AllFC+LSTM"
    retrain_type = args.retrain_type

ds_name = "ucf101_24"
ds_path = f"{ds_root}/UCF101_24/"

save_label = f"{ds_name}_{args.model}_{args.num_f}"
if args.retrain_type != "FC+FinalConv":
    save_label = save_label + "_" + args.retrain_type
if args.long_range:
    save_label = save_label + "_LongRange"

multi_gpu = True
num_devices = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ft = model(num_classes=num_classes, pretrained=True)
model_ft.set_parameter_requires_grad(retrain_type)
params_to_update = model_ft.get_parameter_to_update(debug=False)
model_ft.to_device(device)
if multi_gpu:
    model_ft.parallel_model(device_ids=list(range(num_devices)))

# Path to save and read model parameters
pt_save_dir = join(proj_root, 'model_param', f"{save_label}.pt")

if not args.only_test:
    #================== Train & Evaluate ===================#
    # OPtimizer and Loss fn
    optimizer_ft = torch.optim.SGD(params_to_update, lr=0.0001, momentum=0.9)
    # criterion = nn.NLLLoss()
    if args.model == "r2plus1d_18":
        criterion = nn.CrossEntropyLoss()
    elif args.model == "vgg16lstm":
        criterion = nn.NLLLoss()
    elif args.model == "vgg16lstm_small":
        criterion = nn.CrossEntropyLoss()

    # Dataset and dataloader
    video_datasets = {}
    for x in ['train', 'val']:
        if args.long_range:
            sample_mode = 'long_range_random' if x == 'train' else 'long_range_last'
            num_clips = 3 if x == 'train' else 1
            frame_rate = 6
        else:
            sample_mode = "random" if x=="train" else "fixed"
            num_clips = 1
            frame_rate = 2
        video_datasets[x] = UCF101_24_Dataset(ds_path, args.num_f, sample_mode, num_clips, 
                                                frame_rate, x=='train', testlist_idx=args.testlist_idx)
    print({x: 'Num of clips:{}'.format(len(video_datasets[x])) for x in ['train', 'val']})
    dataloaders = {x: DataLoader(video_datasets[x], batch_size=args.batch_size, shuffle=True, 
                        num_workers=64) for x in ['train', 'val']}

    # Train
    hist = model_ft.train_model(dataloaders, criterion, optimizer_ft, pt_save_dir, args.num_epochs)

#========================== Only validate =========================#
sample_mode = 'long_range_last' if args.long_range else "fixed"
frame_rate = 6 if args.long_range else 2
val_dataset = UCF101_24_Dataset(ds_path, args.num_f, sample_mode, 1, 
                                frame_rate, train=False, testlist_idx=args.testlist_idx)
print('Num of clips:{}'.format(len(val_dataset)))
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, 
                    num_workers=64)
# Validate
y_pred, y_true = model_ft.val_model(val_dataloader, pt_save_dir)

# Draw confusion matrix
class_names = np.asarray(sorted(os.listdir(f'{ds_path}/images')))
cm_fig, _ = plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)
cm_fig.savefig(os.path.join(proj_root, 'visual_res', 'cm', f'{save_label}.png'))
