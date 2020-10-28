import os
from os.path import join, isdir, isfile

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

from utils.ImageShow import *
from utils.GaussianSmoothing import GaussianSmoothing
from utils.ReadingDataset import get_frames, load_model_and_dataset
from process_perturb_res import vis_perturb_res, get_perturb_acc_dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch, torchvision
from skimage.transform import resize
from skimage.filters import gaussian

perturb_areas = [0.05, 0.1, 0.15, 0.5]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ucf101', choices=['epic', 'ucf101', 'cat_ucf'])
    parser.add_argument("--model", type=str, default='r2p1d', choices=['v16l', 'r2p1d', 'r50l'])
    parser.add_argument('--white_bg', action='store_true')
    parser.add_argument('--with_prob', action='store_true')
    # parser.add_argument('--no_save_frames', action='store_true')
    args = parser.parse_args()

    vis_save_dir = os.path.join(proj_root, 'vis_all_for_one', f'{args.dataset}_{args.model}')
    os.makedirs(vis_save_dir, exist_ok=True)

    # all_methods_list = ['g', 'ig', 'sg', 'la', 'grad_cam', 'eb', 'perturb', 'core5', 'core11']
    all_methods_list = ['perturb', 'core5', 'core11']
    vis_classes_list = ['Fencing', 'Diving']

    model_ft, video_dataset = load_model_and_dataset(args.dataset, args.model, testlist_idx=2)
    dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=128)
    video_dataset_dict = {}
    for sample in dataloader:
        x, label, video_name, fidx_tensors = sample
        video_name = video_name[0].split('/')[-1]
        video_dataset_dict[video_name] = (x[0], label[0], fidx_tensors[0])

    if vis_classes_list == []:
        print(f'Since vis_classes_list is [], all videos will be saved.')
        video_names_list = list(video_dataset_dict.keys())
    else:
        video_names_list = []
        for class_name in vis_classes_list:
            for video_name in list(video_dataset_dict.keys()):
                if class_name in video_name:
                    video_names_list.append(video_name)
        if len(video_names_list) == 0:
            raise Exception(f'Something wrong with vis_classes_list.')
        else:
            print(f'video_names_list: {video_names_list}')

    print(f'1. Save frames ...')
    video_frames_dict = {}
    for video_name in tqdm(video_names_list):
        frames_save_dir = os.path.join(vis_save_dir, video_name, 'frames')
        os.makedirs(frames_save_dir, exist_ok=True)

        frames = get_frames(args.dataset, args.model, video_name, video_dataset_dict[video_name][2])    # Tx3x112x112
        video_frames_dict[video_name] = frames  # TxHxWx3
        for fidx, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(frames_save_dir, f'{fidx:02d}.png'))

    exe_res_dir = os.path.join(proj_root, 'exe_res')
    exe_res_pref = f'{args.dataset}_{args.model}'
    for midx, method in enumerate(all_methods_list):
        if 'core' in method:
            exe_res_label = f'{exe_res_pref}_perturb_full_{method}.pt'
        else:
            exe_res_label = f'{exe_res_pref}_{method}_full.pt'
        # if method == 'eb':
        #     exe_res_label = exe_res_label.replace('.pt', '_abs2.pt')
        # if 'perturb' in exe_res_label:
        #     exe_res_label = exe_res_label.replace('.pt', '_summed.pt')

        exe_res = torch.load(os.path.join(exe_res_dir, exe_res_label))['val']
        heatmaps_dict = {res['video_name'].split('/')[-1]: res['mask'] for res in exe_res}

        if 'perturb' in exe_res_label and args.with_prob:
            perturb_res = [res for res in exe_res if res['video_name'].split('/')[-1] in video_names_list]
            video_probs_dict = get_perturb_acc_dict(args.dataset, args.model, perturb_res,
                                            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            video_probs_dict = None

        print(f'{midx+2}. Save vis results for {method}...')
        for video_name in tqdm(video_names_list):
            heatmaps_save_dir = os.path.join(vis_save_dir, video_name, method)
            os.makedirs(heatmaps_save_dir, exist_ok=True)

            heatmaps = heatmaps_dict[video_name].astype(np.float32) # 1xTxHxW
            # heatmaps = (heatmaps - heatmaps.min()) / (heatmaps.max() - heatmaps.min()) # 1xTxHxW

            imgs = voxel_tensor_to_np(video_dataset_dict[video_name][0]) # 3xTxHxW, 0~1

            if 'perturb' not in exe_res_label:
                overlaps = overlap_maps_on_voxel_np(imgs, heatmaps[0]) # 3xTxHxW, 0~1
                plot_voxel_np(imgs, overlaps, title=video_name, 
                                save_path=os.path.join(heatmaps_save_dir, 'sum.png'))
                for fidx in range(overlaps.shape[1]):
                    overlap = (overlaps[:,fidx,:,:]*255).astype(np.uint8).transpose((1,2,0))
                    Image.fromarray(overlap).save(os.path.join(heatmaps_save_dir, f'{fidx:02d}.png'))
            else:
                merged_fig, mats = vis_perturb_res(args.dataset, args.model, video_name, 
                                        heatmaps, frames=video_frames_dict[video_name], 
                                        white_bg=args.white_bg, prob_dict=video_probs_dict)
                Image.fromarray(merged_fig).save(os.path.join(heatmaps_save_dir, 'sum.png'))

                for aidx, overlaps in enumerate(mats[1:-1]):
                    perturb_area = perturb_areas[aidx]
                    overlap_save_label = f'{int(perturb_area*100):02d}'
                    if video_probs_dict != None:
                        area_prob = video_probs_dict[video_name][aidx]
                        overlap_save_label += f'_{int(area_prob*100):02d}'
                    overlap_save_dir = os.path.join(vis_save_dir, video_name, method, overlap_save_label)
                    os.makedirs(overlap_save_dir, exist_ok=True)
                    for fidx, overlap in enumerate(overlaps):
                        Image.fromarray(overlap).save(os.path.join(overlap_save_dir, f'{fidx:02d}.png'))
                        
                overlap_save_dir = os.path.join(vis_save_dir, video_name, method, 'summed')
                os.makedirs(overlap_save_dir, exist_ok=True)
                for fidx, summed_overlap in enumerate(mats[-1]):
                    Image.fromarray(summed_overlap).save(os.path.join(overlap_save_dir, f'{fidx:02d}.png'))



    