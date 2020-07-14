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
sys.path.append("/home/acb11711tx/lzq/VideoPerturb2")
from utils.ConfusionMatrix import plot_confusion_matrix
from utils.CalAcc import AverageMeter, accuracy

crt_dir = os.path.dirname(os.path.realpath(__file__))

def loadTags(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        data = list(reader)
        
    tagName = [r[0] for r in data]
    return tagName, dict(zip(tagName, range(len(tagName))))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["r2plus1d_18", "vgg16lstm"])
parser.add_argument("--num_f", type=int, default=16, choices=[8, 16])
parser.add_argument("--long_range", action='store_true')
parser.add_argument("--num_epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=48)
parser.add_argument("--testlist_idx", type=int, default=2, choices=[1, 2])
parser.add_argument("--only_test", action='store_true')
parser.add_argument("--additional_label", type=str, default="")
parser.add_argument("--retrain_type", type=str, default="FC+FinalConv",
                    choices=["FC+FinalConv", "Full"])
parser.add_argument("--use_amp", action='store_true')
# parser.add_argument("--debug", action='store_true')
args = parser.parse_args()


if args.model == "r2plus1d_18":
    from model_train.r2plus1d import r2plus1d as model
    from datasets.epic_kitchens_dataset_new import EPIC_Kitchens_Dataset
    # retrain_type = "FC+FinalConv"
    retrain_type = args.retrain_type
elif args.model == "vgg16lstm":
    from model_train.vgg16lstm import vgg16lstm as model
    from datasets.epic_kitchens_dataset_vgg16lstm import EPIC_Kitchens_Dataset
    # retrain_type = "FinalConv+AllFC+LSTM"
    retrain_type = args.retrain_type

num_classes = 20

ds_name = "epic"
ds_path = "/home/acb11711tx/lzq/dataset/epic-kitchens/"

save_label = f"{ds_name}_{args.model}_{args.num_f}"
if args.retrain_type != "FC+FinalConv":
    save_label = save_label + "_" + args.retrain_type
if args.long_range:
    save_label = save_label + "_LongRange"
if args.use_amp:
    save_label = save_label + "_amp"
    
save_label = save_label + args.additional_label

multi_gpu = True
num_devices = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ft = model(num_classes=num_classes, with_softmax=True)
model_ft.set_parameter_requires_grad(retrain_type)
params_to_update = model_ft.get_parameter_to_update(debug=False)
model_ft.to_device(device)
if multi_gpu:
    model_ft.parallel_model(device_ids=list(range(num_devices)))

# Path to save and read model parameters
pt_save_dir = join(crt_dir.replace('model_train', 'models'), f"{save_label}.pt")

if not args.only_test:
    #================== Train & Evaluate (for EPIC) ===================#
    # OPtimizer and Loss fn
    optimizer_ft = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    if args.model == "r2plus1d_18":
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.NLLLoss()
    elif args.model == "vgg16lstm":
        criterion = nn.NLLLoss()

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
            frame_rate = 6
        video_datasets[x] = EPIC_Kitchens_Dataset(ds_path, args.num_f, sample_mode, num_clips, 
                                                    frame_rate, x=='train', testlist_idx=args.testlist_idx)
    print({x: 'Num of clips:{}'.format(len(video_datasets[x])) for x in ['train', 'val']})
    dataloaders = {x: DataLoader(video_datasets[x], batch_size=args.batch_size, shuffle=True, 
                        num_workers=128) for x in ['train', 'val']}

    # Train
    hist = model_ft.train_model(dataloaders, criterion, optimizer_ft, pt_save_dir, args.num_epochs, args.use_amp)

#========================== Only validate =========================#
sample_mode = 'long_range_last' if args.long_range else "fixed"
val_dataset = EPIC_Kitchens_Dataset(ds_path, args.num_f, sample_mode, 1, 6, train=False, testlist_idx=args.testlist_idx)
print('Num of clips:{}'.format(len(val_dataset)))
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, 
                    num_workers=128)
# Validate
y_pred, y_true = model_ft.val_model(val_dataloader, pt_save_dir)

# # Draw confusion matrix
tags,tag2ID = loadTags('/home/acb11711tx/lzq/VideoPerturb2/datasets/epic_top20_catName.txt')
class_names = np.asarray(tags)
cm_fig, _ = plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)
cm_fig.savefig(os.path.join(crt_dir.replace('model_train', 'visual_res'), 'cm', f'{save_label}.png'))

# # Save y_pred & y_true
# save_txt = os.path.join(crt_dir.replace('model_train', 'visual_res'), 'cm', f'{save_label}.txt')
# with open(save_txt, "wb") as fp:
#     pickle.dump({'y_pred': y_pred, 'y_true': y_true, 
#                     'class_names': class_names}, fp)