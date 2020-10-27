import os
from os.path import join, isdir, isfile

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

from utils.ImageShow import *
from utils.ReadingDataset import load_model_and_dataset, getTagScore, loadTags
from process_perturb_res import vis_perturb_res

from visual_meth.integrated_grad import integrated_grad
from visual_meth.gradients import gradients
from visual_meth.perturbation_area import video_perturbation
from visual_meth.grad_cam import grad_cam
from visual_meth.smooth_grad import smooth_grad
from visual_meth.excitation_backprop import excitation_bp
from visual_meth.guided_backprop import guided_bp
from visual_meth.linear_approximation import linear_appr

import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from tqdm import tqdm
import math
import time
import csv
import copy

def single_worker(gpu, args):
    if args.vis_method == 'perturb':
        if args.model == 'r2p1d' or args.model == "r50l":
            step = 7
            sigma = 11
        elif args.model == 'v16l':
            step = 14
            sigma = 23
    batch_size = args.batch_size

    torch.manual_seed(0)

    model_ft, video_datasets = load_model_and_dataset(args.dataset, args.model, args.phase_set)
    model_ft = model_ft.eval()  # important!
    model_ft.cuda(gpu)
    # model_ft = DDP(model_ft, device_ids=[gpu])

    # Initialize the dataset and dataloader
    dataloaders = {x: DataLoader(video_datasets[x], batch_size=batch_size, shuffle=False, 
                        num_workers=128) for x in args.phase_set}
    print({x: 'Num of batches:{}'.format(len(dataloaders[x])) for x in args.phase_set})

    if args.dataset == 'epic':
        tags,tag2ID = loadTags(f'{proj_root}/datasets/epic_top20_catName.txt')
    elif args.dataset == 'ucf101':
        tags,tag2ID = loadTags(f'{proj_root}/datasets/ucf101_24_catName.txt')

    if args.visualize:
        plot_save_path = f"{proj_root}/visual_res/{args.save_label}"
        os.makedirs(plot_save_path, exist_ok=True)

    # res_buf = {'train': [], 'val': []}
    for phase in args.phase_set:
        res_buf = {'train': [], 'val': []}
        for samples in dataloaders[phase]:
            # x: 1x3x16x112x112; label: 1; output mask: 
            # 1x1x16x112x112; fidx_tensors: 1x16;
            x, labels, seg_names, fidx_tensors = samples
            x = x.cuda(gpu)
            labels = labels.to(dtype=torch.long).cuda(gpu)
            # print(x.shape, x.device)

            y = model_ft(x)
            lowest_probs, lowest_labels = torch.min(y, dim=1)
            # for bidx in range(ymin_labels.shape[0]):
            #     print(f'{ymin_labels[bidx]}: {tags[ymin_labels[bidx]]}')
            # print(f'{labels.shape}, {ymin_labels.shape}')

            device = x.device

            if args.vis_method == 'g':
                res = gradients(x, labels, model_ft, device, multiply_input=False, polarity='both')
                heatmaps_np = res.numpy()   # Nx1xTxHxW
            elif args.vis_method == 'ig':
                res = integrated_grad(x, labels, model_ft, device, steps=25, polarity='both')
                heatmaps_np = res.numpy()   # Nx1xTxHxW
            elif args.vis_method in ['sg', 'sg2', 'sg_var']:
                variant_dict = {'sg': None, 'sg2': 'square', 'sg_var': 'variance'}
                variant = variant_dict[args.vis_method]
                res = smooth_grad(x, labels, model_ft, device, nsamples=25, variant=variant)
                heatmaps_np = res.numpy()   # Nx1xTxHxW
            elif args.vis_method == 'grad_cam':
                if args.model == 'r2p1d':
                    layer_name = ['layer4']
                    # layer_name = ['layer3']
                elif args.model == 'v16l':
                    layer_name = ['pool5']
                elif args.model == 'r50l':
                    layer_name = ['6']
                res = grad_cam(x, labels, model_ft, args.model, device, layer_name=layer_name, norm_vis=True)
                heatmaps_np = res.numpy()   # Nx1xTxHxW
                # heatmaps_np = 1 - (1 - heatmaps_np ** 2.0) ** 2.4
            elif args.vis_method == 'eb':   # Cannot support RNN
                if args.model == 'r2p1d':
                    layer_name = ['layer4']
                elif args.model == 'r50l':
                    layer_name = ['6']
                else:
                    raise Exception(f"Excitation BP supports only R(2+1)D now. Given {args.model}.")
                res = excitation_bp(x, labels, model_ft, args.model, device, layer_name=layer_name, norm_vis=True)
                heatmaps_np = res.numpy()   # Nx1xTxHxW
            elif args.vis_method == 'gbp':
                res = guided_bp(x, labels, model_ft)
                heatmaps_np = res.numpy()
            elif args.vis_method == 'la':
                res = linear_appr(x, labels, model_ft)
                heatmaps_np = res.numpy()
            elif args.vis_method == 'perturb':
                sigma = 11 if x.shape[-1] == 112 else 23
                if args.lowest_label:
                    res = video_perturbation(
                        model_ft, x, lowest_labels, areas=args.areas, sigma=sigma, 
                        max_iter=args.perturb_niter, variant="preserve",
                        gpu_id=gpu, print_iter=200, perturb_type="blur",
                        with_core=args.perturb_withcore, core_num_keyframe=args.perturb_num_keyframe)[0]
                else:
                    res = video_perturbation(
                            model_ft, x, labels, areas=args.areas, sigma=sigma, 
                            max_iter=args.perturb_niter, variant="preserve",
                            gpu_id=gpu, print_iter=200, perturb_type="blur",
                            with_core=args.perturb_withcore, core_num_keyframe=args.perturb_num_keyframe)[0]
                # print(res.shape)
                heatmaps_np = res.numpy()   # NxAx1xTxHxW
                # print(heatmaps_np.shape)
                
            for bidx in range(len(seg_names)):
                seg_name = copy.deepcopy(seg_names[bidx].split("/")[-1])
                heatmap = heatmaps_np[bidx].astype('float16')
                fidxs = copy.deepcopy(fidx_tensors[bidx].detach().cpu().numpy())
                fidxs = fidxs.astype('uint16')

                res_buf[phase].append({"video_name": seg_name, "mask": heatmap, "fidx": fidxs})
                print(seg_names[bidx])

                if args.visualize:
                    inp_np = voxel_tensor_to_np(x[bidx].detach().cpu())   # 1 x num_f x 224 224
                    if args.vis_method == 'perturb':
                        merged_fig, _ = vis_perturb_res(args.dataset, args.model, seg_name, 
                                                    heatmaps_np[bidx], frame_idx=fidxs, white_bg=False)
                        Image.fromarray(merged_fig).save(os.path.join(args.plot_save_path, seg_name+'.jpg'))
                    else:
                        if args.vis_method == 'grad_cam' or args.vis_method == 'eb':
                            heatmap_np = overlap_maps_on_voxel_np(inp_np, heatmaps_np[bidx,0])
                        else:
                            heatmap_np = heatmaps_np[bidx].repeat(3, axis=0)      # 3 x num_f x 224 224
                        plot_voxel_np(inp_np, heatmap_np, title=seg_name, 
                                        save_path=os.path.join(args.plot_save_path, seg_name+'.jpg') )

        # res_save_name = f'{args.save_label}_{phase}.pt'
        # torch.save(res_buf, join(args.res_save_path, res_save_name))
        # print(f'Phase:{phase} saved.', res_save_name)
    res_save_name = f'{args.save_label}.pt'
    torch.save(res_buf, join(args.res_save_path, res_save_name))
    print(f'All saved', res_save_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['epic', 'ucf101', 'cat_ucf'])
    parser.add_argument("--model", type=str, choices=['r2p1d', 'v16l', 'r50l'])
    parser.add_argument("--vis_method", type=str, 
                        choices=['g', 'ig', 'sg', 'sg2', 'grad_cam', 'perturb', 'eb', 'gbp', 'la'])
    parser.add_argument("--only_test", action="store_true")
    parser.add_argument("--only_train", action="store_true")
    # parser.add_argument("--num_gpu", type=int, default=-1)
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--extra_label", type=str, default="")
    parser.add_argument("--retrain_type", type=str, default="full", choices=["full", "half"])
    parser.add_argument("--lowest_label", action='store_true')
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--perturb_niter", type=int, default=1000)
    parser.add_argument("--perturb_withcore", action='store_true')
    parser.add_argument("--perturb_num_keyframe", type=int, default=5)
    # parser.add_argument("--perturb_spatial_size", type=int, default=11)

    parser.add_argument("--master_addr", type=str, default="127.0.1.1")
    parser.add_argument("--master_port", type=str, default="29501")

    args = parser.parse_args()

    assert args.only_test & args.only_train == False, "only_test and only_train cannot be true together!"
    # args.phase_set = ["val"] if args.only_test else ["val", "train"]
    # args.phase_set = ["train"] if args.only_train else ["val", "train"]
    args.phase_set = ["val", "train"]
    if args.only_test:
        args.phase_set = ["val"]
    if args.only_train:
        args.phase_set = ["train"]

    if args.dataset == 'cat_ucf':
        args.phase_set = ["val"]

    print(args.phase_set)

    # Set path to save masks generated by perturbations
    save_label = f"{args.dataset}_{args.model}_{args.vis_method}_{args.retrain_type}"
    if args.lowest_label:
        save_label = save_label + "_lowest"
    if args.perturb_withcore:
        save_label = save_label + "_core" + f"{args.perturb_num_keyframe}"
    if args.extra_label != "":
        save_label = save_label + f"_{args.extra_label}"
    args.save_label = save_label
    print(save_label)

    res_save_path = f"{proj_root}/exe_res"
    os.makedirs(res_save_path, exist_ok=True)
    args.res_save_path = res_save_path

    if args.visualize:
        plot_save_path = f"{proj_root}/visual_res/{save_label}"
        os.makedirs(plot_save_path, exist_ok=True)
        args.plot_save_path = plot_save_path

    if args.vis_method == 'perturb':
        args.areas = [0.05, 0.1, 0.15, 0.5]
        if args.dataset == 'cat_ucf':
            args.areas = [0.02, 0.05, 0.1]

    # print(f'Use {num_devices} GPUs.')
    # mp.spawn(main_worker, nprocs=num_devices, args=(args,))
    # print('*** This is the end of the multiprocessing ***')
    single_worker(0, args)

    # all_res = {'train': [], 'val': []}
    # for phase in args.phase_set:
    #     for device_id in range(num_devices):
    #         res_save_name = f'{args.save_label}_gpu{device_id}_{phase}.pt'
    #         res_buf = torch.load(join(args.res_save_path, res_save_name))
    #         all_res[phase] += res_buf[phase]
    # phase_label = ""
    # if args.only_test:
    #     phase_label = "_test"
    # if args.only_train:
    #     phase_label = "_train"
    # res_save_name = f'{args.save_label}{phase_label}.pt'
    # torch.save(all_res, join(args.res_save_path, res_save_name))
    # print(f'Saved all samples as {res_save_name}.')

    # for phase in args.phase_set:
    #     for device_id in range(num_devices):
    #         res_save_name = f'{args.save_label}_{phase}.pt'
    #         os.remove(join(args.res_save_path, res_save_name))
    # print(f'Deleted all samples saved by each GPU.')