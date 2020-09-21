import os
from os.path import join, isdir, isfile

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

from utils.ImageShow import *
from utils.GaussianSmoothing import GaussianSmoothing
from utils.CalAcc import process_activations
from utils.ReadingDataset import get_frames, load_model_and_dataset, loadTags

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cv2
import math
import numpy as np
import csv
from tqdm import tqdm
from PIL import Image
import torch, torchvision
from skimage.transform import resize
from skimage.filters import gaussian

from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ucf101', choices=['epic', 'ucf101', 'cat_ucf'])
    parser.add_argument("--model", type=str, default='r2p1d', choices=['v16l', 'r2p1d', 'r50l'])
    parser.add_argument("--vis_method", type=str, default='perturb', choices=['g', 'ig', 'sg', 'sg2', 'grad_cam', 'eb', 'perturb'])
    # parser.add_argument('--visualize', action='store_true')
    # parser.add_argument('--only_test', action='store_true')
    # parser.add_argument('--only_train', action='store_true')    
    parser.add_argument("--phase", type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--extra_label', type=str, default="")  
    args = parser.parse_args()

    if args.dataset == 'epic':
        tags,tag2ID = loadTags(f'{proj_root}/datasets/epic_top20_catName.txt')
    elif args.dataset == 'ucf101':
        tags,tag2ID = loadTags(f'{proj_root}/datasets/ucf101_24_catName.txt')

    res_label = f'{args.dataset}_{args.model}_{args.vis_method}_full{args.extra_label}'
    if args.vis_method == 'perturb':
        res_label += '_summed'

    res_dir = os.path.join(proj_root, 'exe_res', res_label+'.pt')
    res_dict = torch.load(res_dir)

    summed_res = {'train': list(), 'val': list()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ft, video_dataset = load_model_and_dataset(args.dataset, args.model, args.phase)
    model_ft.to(device)
    model_ft.eval()
    dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=128)

    prob_dict = {}
    label_dict = {}
    for sample in dataloader:
        x, labels, video_names, fidx_tensors = sample

        bs = x.shape[0]
        clip_tensor = x.to(device)  # NxCxTxHxW
        y = model_ft(clip_tensor)    #Nx num_classes

        labels = labels.to(torch.long)
        probs, pred_labels, pred_label_probs = process_activations(y, labels, softmaxed=True)   # prob: N
        
        for bidx in range(bs):
            v_name = video_names[bidx].split('/')[-1]
            prob_dict[v_name] = probs[bidx].detach().item()
            label_dict[v_name] = labels[bidx].detach().item()

    summed_masks_by_label = [[] for i in range(len(tags))]
    probs_by_label = [[] for i in range(len(tags))]
    for res in tqdm(res_dict[args.phase]):
        video_name = res["video_name"].split('/')[-1]
        mask = res["mask"].astype(np.float)     #1xTxHxW

        video_label = label_dict[video_name]
        video_prob = prob_dict[video_name]
        summed_mask = torch.from_numpy(mask).mean(-1).mean(-1) # 1xT
        summed_mask /= summed_mask.sum(axis=1, keepdims=True)

        summed_masks_by_label[video_label].append(summed_mask) 
        probs_by_label[video_label].append(video_prob)

    plt_col = 4
    plt_row = math.ceil(len(tags) / plt_col)
    fig, axes = plt.subplots(plt_row, plt_col, figsize=(4*plt_col, 3*plt_row))

    plt.subplots_adjust(left=0.05,
                    bottom=0.05, 
                    right=0.95, 
                    top=0.95, 
                    wspace=0.15, 
                    hspace=0.2)

    for i, tag in enumerate(tags):
        print(tag, len(summed_masks_by_label[i]))
        summed_masks_by_label[i] = torch.cat(summed_masks_by_label[i], dim=0).numpy()
        probs_by_label[i] = np.array(probs_by_label[i])

        row_idx = i // plt_col
        col_idx = i % plt_col
        ax = axes[row_idx, col_idx]
        x = np.arange(16)
        y_mean = summed_masks_by_label[i].mean(axis=0)
        y_std = summed_masks_by_label[i].std(axis=0)
        prob_mean = probs_by_label[i].mean()
        prob_std = probs_by_label[i].std()

        ax.plot(x, y_mean, alpha=0.5, color='red', label='mean', linewidth = 1.0)
        ax.fill_between(x, y_mean-y_std, y_mean+y_std, color='#888888', alpha=0.4)
        ax.set_title(f"{tag}  {prob_mean:.3f}/{prob_std:.3f}", fontsize=12)
        ax.set_ylim([0.0, 0.12])

    fig.savefig(f'{args.dataset}_{args.model}_{args.vis_method}_{args.phase}.png')
