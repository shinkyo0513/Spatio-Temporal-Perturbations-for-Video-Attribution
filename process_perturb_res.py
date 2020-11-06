import os
from os.path import join, isdir, isfile

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

from utils.ImageShow import *
from utils.GaussianSmoothing import GaussianSmoothing
from utils.ReadingDataset import get_frames, load_model_and_dataset

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

def get_perturb_acc_dict (dataset_name, model_name, perturb_res_list, device):
    from visual_meth.perturbation_area import Perturbation
    from utils.CalAcc import process_activations
    from torch.utils.data import Dataset, DataLoader

    model_ft, video_dataset = load_model_and_dataset(dataset_name, model_name)
    model_ft.to(device)
    model_ft.eval()
    dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=128)

    clip_tensor_dict = {}
    clip_label_dict = {}
    for sample in dataloader:
        x, label, video_name, fidx_tensors = sample
        video_name = video_name[0].split('/')[-1]
        clip_tensor_dict[video_name] = x[0]
        clip_label_dict[video_name] = label[0]

    # print(clip_tensor_dict.keys())

    prob_dict = {}
    for res in tqdm(perturb_res_list):
        video_name = res["video_name"]
        masks = res["mask"].astype(np.float)     #Ax1xTxHxW
        # print(masks.shape)
        fids = res["fidx"]

        clip_tensor = clip_tensor_dict[video_name].to(device)  # CxTxHxW
        pmt_inp = clip_tensor.transpose(0,1).contiguous() # TxCxHxW
        # pmt_inp = pmt_inp.view(1*16, *pmt_inp.shape[2:])  # N*T x CxHxW
        perturbation = Perturbation(pmt_inp, num_levels=8, type="blur").to(device)

        masks_tensor = torch.from_numpy(masks.astype(np.float32)).transpose(1,2)    #AxTx1xHxW
        masks_tensor = masks_tensor.view(-1, *masks_tensor.shape[2:])   #A*T x1xHxW
        masks_tensor = masks_tensor.to(device)
        perturb_x = perturbation.apply(masks_tensor)  # A*T x 1xCxHxW
        perturb_x = perturb_x.view(-1, 16, *perturb_x.shape[2:]) # AxTxCxHxW
        perturb_x = perturb_x.transpose(1, 2).contiguous()   # AxCxTxHxW
        y = model_ft(perturb_x)    #Ax num_classes

        label = clip_label_dict[video_name]
        label = torch.tensor([label]).to(torch.long)
        prob, pred_label, pred_label_prob = process_activations(y, label, softmaxed=True)   # prob: A
        # print(f"{video_name}: {prob[0]:.3f}")
        prob_dict[video_name] = prob.detach().cpu().numpy()
    
    return prob_dict

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
    merged_mat = []
    if gap > 0:
        shape = list(mat[0].shape)
        shape[dim] = gap
        for i in range(len(mat) - 1):
            merged_mat.append(np.concatenate(
                [mat[i], np.ones(shape, dtype=np.uint8) * 255], axis=dim
            ))
        merged_mat.append(mat[-1])
    return np.concatenate(merged_mat, axis=dim)

def vis_perturb_res (dataset, model, video_name, masks, frame_index=None, frames=None, white_bg=False, with_text=True, prob_dict=None):
    video_name_regu = video_name.split("/")[-1]
    if isinstance(frame_index, np.ndarray):
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
    # print(len(merged_lines))
    if with_text:
        if prob_dict != None:
            prob = prob_dict[video_name]
        for a_idx, area_line in enumerate(merged_lines[1:-1]):
            shape = list(area_line.shape)
            shape[0] = 15
            white_bar = np.ones(shape, dtype=np.uint8) * 255
            area_line = np.concatenate([white_bar, area_line], axis=0)
            if prob_dict != None:
                area_line = cv2.putText(area_line, f'Area: {real_areas[a_idx]:.2f}; Prob: {prob[a_idx]:.3f}', 
                                    (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
            else:
                area_line = cv2.putText(area_line, f'Area: {real_areas[a_idx]:.2f}', 
                                    (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
            merged_lines[1+a_idx] = area_line
    # print(len(merged_lines))
    merged_fig = merge(merged_lines, 0, gap=8)
    return merged_fig, mats

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
    parser.add_argument("--dataset", type=str, default='ucf101', choices=['epic', 'ucf101', 'cat_ucf'])
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
    if args.visualize:
        vis_save_path = os.path.join(proj_root, 'visual_res', res_label)
        os.makedirs(vis_save_path, exist_ok=True)

    perturb_res_dir = os.path.join(proj_root, 'exe_res', res_label+'.pt')
    perturb_res = torch.load(perturb_res_dir)

    summed_res = {'train': list(), 'val': list()}

    if args.specify_video == None:
        for phase in perturb_res.keys():
        # for phase in ['val']:
            if phase == 'val':
                prob_dict = get_perturb_acc_dict(args.dataset, args.model, perturb_res[phase], 
                                    torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            else:
                prob_dict = None
            # prob_dict = get_perturb_acc_dict(args.dataset, args.model, perturb_res[phase], 
            #                         torch.device("cuda" if torch.cuda.is_available() else "cpu"))
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
                    
                    merged_fig, mats = vis_perturb_res(args.dataset, args.model, video_name_regu, 
                                                masks, frames=frames, white_bg=args.white_bg, prob_dict=prob_dict)
                    Image.fromarray(merged_fig).save(os.path.join(vis_save_path, f"{video_name_regu}.jpg"))
                    print(f'Saved {video_name_regu}')

                summed_masks = sum_masks(masks)
                summed_res[phase].append({'video_name': video_name, 
                                          'mask': summed_masks.astype('float16'), 
                                          'fidx': fids})
        torch.save(summed_res, perturb_res_dir.replace('.pt', '_summed.pt'))
        print('Finished.')