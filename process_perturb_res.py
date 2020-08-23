import os
from os.path import join, isdir, isfile

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

from utils.ImageShow import *
from utils.GaussianSmoothing import GaussianSmoothing

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch, torchvision
from skimage.transform import resize
from skimage.filters import gaussian

def standardize_epic_video_name(name):
    video_name = name.split('/')[-1]
    prefix = video_name.split('-')[0].split('_')
    return '/'.join([prefix[0], '_'.join(prefix[:2]), video_name])

def standardize_ucf101_video_name(name):
    classification = name.split('_')[1]
    return '/'.join(['images', classification, name])

def get_frames(dataset_name, model_name, video_name, fids):
    if dataset_name == 'epic':
        vname = standardize_epic_video_name(video_name)
        root_path = f'{ds_root}/epic/seg_train'
        video_stf = int(sorted(os.listdir(os.path.join(root_path, vname)))[0][-14:-4])
        frames = [Image.open(os.path.join(root_path, vname, 
                        f'frame_{fid + video_stf:010d}.jpg')) for fid in fids]
    elif dataset_name == 'ucf101':
        root_path = f'{ds_root}/UCF101_24'
        frames = [Image.open(os.path.join(root_path, standardize_ucf101_video_name(video_name),
                        format(fid + 1, '05d') + '.jpg')) for fid in fids]
    else:
        print('ERROR!')
        return
    
    if model_name == 'r2p1d' or model_name == 'r50l':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((112, 112)),
        ])
    elif model_name == 'v16l':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((240, 320)),
            torchvision.transforms.CenterCrop((224, 224))
        ])
    else:
        print('ERROR!')
        return
    
    return np.stack([np.array(transform(frame)) for frame in frames])

JET_CMAP = plt.get_cmap('jet')
def mask_overlap(frames, masks, hm_flag, white_bg=False, frame_wise_norm=False):
    BLACK_BG = Image.fromarray(np.zeros(frames.shape[1:], np.uint8))
    WHITE_BG = Image.fromarray(255 * np.ones(frames.shape[1:], np.uint8))
    overlaps = []

    for i, (frame, mask) in enumerate(zip(frames, masks)):
        h, w, _ = frame.shape
        mask_resized = resize(mask, (w, h))
        if hm_flag and frame_wise_norm:
            mask_resized /= mask_resized.max()
        mask_resized = 1 - (1 - mask_resized ** 2.0) ** 2.4
        if hm_flag:
            bg = Image.fromarray((JET_CMAP(mask_resized) * 255.0).astype(np.uint8))
            msk = Image.fromarray((mask_resized * 255).astype(np.uint8))
        else:
            if white_bg:
                bg = WHITE_BG
            else:
                bg = BLACK_BG
            msk = Image.fromarray((255 - mask_resized * 255).astype(np.uint8))
        img = Image.fromarray(frame)
        overlap = np.array(Image.composite(bg, img, msk))[..., :3]
        overlaps.append(overlap)

    return overlaps

def merge(mat, dim, gap=0):
    if type(mat) != list:
        assert(len(mat.shape) == 4)
        mat = list(mat)        
    if gap > 0:
        shape = list(mat[0].shape)
        shape[dim] = gap
        for i in range(len(mat) - 1):
            mat[i] = np.concatenate(
                [mat[i], np.ones(shape, dtype=np.uint8) * 255], axis=dim
            )
    return np.concatenate(mat, axis=dim)

def vis_perturb_res (dataset, model, video_name, masks, frame_index, white_bg=False, with_text=True):
    video_name_regu = video_name.split("/")[-1]
    frames = get_frames(dataset, model, video_name_regu, frame_index)

    num_area, nch, num_f, nrow, ncol = masks.shape
    real_areas = [masks[a_idx].mean() for a_idx in range(num_area)]

    overlaps = []
    for a_idx in range(num_area):
        area_masks = [masks[a_idx, 0,f_idx] for f_idx in range(num_f)]
        area_overlaps = mask_overlap(frames, area_masks, hm_flag=0, white_bg=white_bg)
        overlaps.append(area_overlaps)

    summed_masks = np.sum(masks, axis=0)  # 1xTxHxW
    summed_masks = [gaussian(summed_masks[0,fidx,...], sigma=10) for fidx in range(num_f)]
    summed_masks = np.stack(summed_masks, axis=0)  # TxHxW
    summed_masks /= summed_masks.max()  # TxHxW
    summed_masks = list(summed_masks)
    sum_overlaps = mask_overlap(frames, summed_masks, hm_flag=1)
    overlaps.append(sum_overlaps)

    mats = [frames, ] + overlaps
    merged_lines = [merge(mat, 1, gap=5) for mat in mats]
    if with_text:
        for a_idx, area_line in enumerate(merged_lines[1:-1]):
            shape = list(area_line.shape)
            shape[0] = 15
            white_bar = np.ones(shape, dtype=np.uint8) * 255
            area_line = np.concatenate([white_bar, area_line], axis=0)
            area_line = cv2.putText(area_line, f'{real_areas[a_idx]:.2f}', 
                                    (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
            merged_lines[1+a_idx] = area_line
    merged_fig = merge(merged_lines, 0, gap=8)
    return merged_fig

def sum_masks (masks, norm=True):
    # masks: Ax1xTxHxW
    num_f = masks.shape[2]
    summed_masks = np.sum(masks, axis=0)  # 1xTxHxW
    summed_masks = [gaussian(summed_masks[0,fidx,...], sigma=10) for fidx in range(num_f)]
    summed_masks = np.stack(summed_masks, axis=0)  # TxHxW
    summed_masks = np.expand_dims(summed_masks, axis=0) # 1xTxHxW
    if norm:
        summed_masks /= summed_masks.max()  # 1xTxHxW
    return summed_masks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ucf101', choices=['epic', 'ucf101'])
    parser.add_argument("--model", type=str, default='r2p1d', choices=['v16l', 'r2p1d', 'r50l'])
    parser.add_argument('--white_bg', action='store_true')
    parser.add_argument("--specify_video", type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--only_train', action='store_true')     
    parser.add_argument('--extra_label', type=str, default="")  
    args = parser.parse_args()

    phase_label = ""
    if args.only_test:
        phase_label = "_test"
    if args.only_train:
        phase_label = "_train"

    res_label = f'{args.dataset}_{args.model}_perturb_full{args.extra_label}{phase_label}'
    vis_save_path = os.path.join(proj_root, 'visual_res', res_label)
    os.makedirs(vis_save_path, exist_ok=True)

    perturb_res_dir = os.path.join(proj_root, 'exe_res', res_label+'.pt')
    perturb_res = torch.load(perturb_res_dir)

    # res_label = f'{args.dataset}_{args.model}_perturb_full'
    # perturb_res_dir_test = os.path.join(proj_root, 'exe_res', res_label+'_test.pt')
    # perturb_res_test = torch.load(perturb_res_dir_test)
    # perturb_res_dir_train = os.path.join(proj_root, 'exe_res', res_label+'_train.pt')
    # perturb_res_train = torch.load(perturb_res_dir_train)

    # perturb_res = {'train': perturb_res_train['train'], 'val': perturb_res_test['val']}
    # perturb_res_dir = os.path.join(proj_root, 'exe_res', res_label+'.pt')
    # torch.save(perturb_res, perturb_res_dir)
    # print('Saved.')

    summed_res = {'train': list(), 'val': list()}

    if args.specify_video == None:
        for phase in perturb_res.keys():
            for res in tqdm(perturb_res[phase]):
                video_name = res["video_name"]
                masks = res["mask"].astype(np.float)     #Ax1xTxHxW
                # masks = np.concatenate((masks[:-2], masks[-1:]), axis=0)
                fids = res["fidx"]

                video_name_regu = video_name.split("/")[-1]

                # Visualization Part
                if args.visualize:
                    frames = get_frames(args.dataset, args.model, video_name_regu, fids)

                    num_area, nch, num_f, nrow, ncol = masks.shape
                    real_areas = [masks[a_idx].mean() for a_idx in range(num_area)]

                    overlaps = []
                    for a_idx in range(num_area):
                        area_masks = [masks[a_idx, 0,f_idx] for f_idx in range(num_f)]
                        area_overlaps = mask_overlap(frames, area_masks, hm_flag=0, white_bg=args.white_bg)
                        overlaps.append(area_overlaps)

                    summed_masks = np.sum(masks, axis=0)  # 1xTxHxW
                    summed_masks = [gaussian(summed_masks[0,fidx,...], sigma=10) for fidx in range(num_f)]
                    # summed_masks = np.stack(summed_masks, axis=0).unsqueeze(0)  # 1xTxHxW
                    sum_overlaps = mask_overlap(frames, summed_masks, hm_flag=1)
                    overlaps.append(sum_overlaps)

                    mats = [frames, ] + overlaps
                    merged_lines = [merge(mat, 1, gap=5) for mat in mats]
                    merged_fig = merge(merged_lines, 0, gap=8)
                    
                    merged_fig = vis_perturb_res(args.dataset, args.model, video_name_regu, 
                                                masks, fids, white_bg=args.white_bg)
                    Image.fromarray(merged_fig).save(os.path.join(vis_save_path, f"{video_name_regu}.jpg"))
                    print(f'Saved {video_name_regu}')

                summed_masks = sum_masks(masks)
                summed_res[phase].append({'video_name': video_name, 
                                          'mask': summed_masks.astype('float16'), 
                                          'fidx': fids})
        torch.save(summed_res, perturb_res_dir.replace('.pt', '_summed.pt'))
        print('Finished.')