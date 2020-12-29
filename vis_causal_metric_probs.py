import os
from os.path import join, isdir, isfile

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

from utils.CausalMetric import plot_causal_metric_curve, auc
from utils.ReadingDataset import get_frames, load_model_and_dataset
from process_perturb_res import vis_perturb_res, get_perturb_acc_dict
# from perturb.perturb_utils import *
from utils.ImageShow import *

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import math
import numpy as np
import pandas as pd
from skimage import transform

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ucf101", 
                                choices=["ucf101", "epic"])
parser.add_argument("--model", type=str, default="r2p1d",
                                choices=["v16l", "r2p1d", "r50l"])
# parser.add_argument("--vis_method", type=str, 
#                                 choices=["g", "ig", "sg", "sg2", "grad_cam", "perturb", "random", "eb", "la", "gbp"])  
parser.add_argument("--mode", type=str, default="ins", choices=["ins", "del", "both"])
parser.add_argument("--order", type=str, default="most_first", choices=["most_first", "least_first"])
parser.add_argument("--new_size", type=int, default=16)
parser.add_argument('--extra_label', type=str, default="")   
parser.add_argument('--save_perturbed_video', action="store_true")
parser.add_argument('--perturb_ratio', type=float, default=0.2)
args = parser.parse_args()

num_score = 128 + 1
x_coords = np.arange(num_score) / num_score
# plt.rcParams.update({'font.size': 16})
fig = plt.figure(figsize=(8,8))
plt.subplot(111)
plt.xlim(0, 1.0)
plt.ylim(0, 1.05)
plt.xlabel('Pixels Perturbed (%)', fontsize=20)
plt.ylabel('Probability (%)', fontsize=20)

frame_transform = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
        ])

video_name = "v_Skijet_g04_c02"
vis_methods = ["grad_cam", "random"]
colors = ["orange", "blue"]

final_save_label = f"{args.dataset}_{args.model}_{args.mode}_{vis_methods[0]}_{vis_methods[1]}"
final_save_path = join(proj_root, "cm_probs_vis", final_save_label)
os.makedirs(final_save_path, exist_ok=True)

for midx, vis_method in enumerate(vis_methods):
    method_save_label = f"{args.dataset}_{args.model}_{args.mode}_{vis_method}"
    if args.extra_label != "":
        method_save_label += f"{args.extra_label}"
    if args.new_size != None:
        method_save_label += f"_{args.new_size}"

    probs_save_path = join(proj_root, "cm_probs", method_save_label+".csv")
    probs_df = pd.read_csv(probs_save_path)
    probs_dict = {}
    for ridx, row in probs_df.iterrows():
        row_els = list(dict(row).values())
        probs_dict[row_els[0]] = row_els[1:]

    video_probs = probs_dict[video_name]
    plt.plot(x_coords, video_probs, color=colors[midx], linestyle='solid', linewidth=2, label=f"{vis_method}: {video_name[2:]}")

    probs_array = np.array(list(probs_dict.values()))
    avg_probs = probs_array.mean(axis=0)
    plt.plot(x_coords, avg_probs, color=colors[midx], linestyle='dashed', linewidth=2, label=f"{vis_method}: Average")

    plt.axvline(args.perturb_ratio, 0, 1, color='gray', linewidth=2)

    if vis_method != 'random':
        res_save_path = join(proj_root, 'exe_res', f"{args.dataset}_{args.model}_{vis_method}_full.pt")
        res_dic_lst = torch.load(res_save_path)['val']
        for res_dic in res_dic_lst:
            if res_dic["video_name"].split("/")[-1] == video_name:
                masks_tensor = torch.from_numpy(res_dic["mask"].astype('float32')).transpose(1,0)  # Tx1xH'xW'
                fidxs = res_dic["fidx"]
                break
        frames = get_frames(args.dataset, args.model, video_name, fidxs)  # Tx3xHxW
        frames_tensor = torch.stack([frame_transform(Image.fromarray(frame)) for frame in frames], dim=0)
        nt, nch, nrow, ncol = frames_tensor.shape
    else:
        masks_tensor = torch.randn((nt, 1, nrow, ncol))

    if masks_tensor.shape[-1] != frames_tensor.shape[-1]:
        masks_tensor = F.interpolate(masks_tensor, size=(nrow, ncol), mode="bilinear")

    assert nrow % args.new_size == 0
    ks = nrow // args.new_size
    k = torch.ones((1, 1, ks, ks)) / (ks*ks)
    small_masks_tensor = F.conv2d(masks_tensor, k, stride=ks, padding=0)    # Tx1x sH x sW
    # T*sH*sW
    sal_order = small_masks_tensor.reshape(-1).argsort(dim=-1, descending=True)

    perturb_topk = int(sal_order.shape[0] * args.perturb_ratio)
    pos = sal_order[:perturb_topk]

    if args.mode == "del":
        pmasks = torch.ones(nt, 1, args.new_size, args.new_size)
        pmasks.reshape(-1)[pos] = 0
    elif args.mode == "ins":
        pmasks = torch.zeros(nt, 1, args.new_size, args.new_size)
        pmasks.reshape(-1)[pos] = 1
    masks_tensor = F.interpolate(pmasks, size=(nrow, ncol), mode='nearest')  # T x 1 x H x W
    pframes_tensor = frames_tensor * masks_tensor + torch.zeros_like(frames_tensor) * (1 - masks_tensor)

    pframe_save_path = join(final_save_path, f"{video_name}_{vis_method}")
    os.makedirs(pframe_save_path, exist_ok=True)

    for it in range(nt):
        pframe_tensor = pframes_tensor[it]
        pframe_np = img_tensor_to_np(pframe_tensor) # 0~1
        pframe_forshow = (pframe_np * 255).astype(np.uint8).transpose(1,2,0)
        Image.fromarray(pframe_forshow).save(join(pframe_save_path, f'{it:02d}.pdf'))

# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=16)
plt.legend(loc='upper left', fontsize=18)
plt.savefig(join(final_save_path, "comparison.pdf"))

    

