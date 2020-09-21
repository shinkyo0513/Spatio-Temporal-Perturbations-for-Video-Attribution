import os
from os.path import join, isdir, isfile

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

from utils.CausalMetric import CausalMetric, plot_causal_metric_curve
from utils.CausalMetric import auc as cal_auc
# from perturb.perturb_utils import *
from utils.ImageShow import *

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import math
import numpy as np
from skimage import transform

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ucf101", 
                                    choices=["ucf101", "epic"])
parser.add_argument("--model", type=str, choices=["v16l", "r2p1d", "r50l"])
parser.add_argument("--vis_method", type=str, choices=["g", "ig", "sg", "sg2", "grad_cam", "perturb"])  
parser.add_argument("--mode", type=str, default="ins", 
                                choices=["ins", "del", "both"])
parser.add_argument("--multi_gpu", action='store_true')
parser.add_argument("--keep_ratio", type=float, default=1.0)
parser.add_argument("--new_size", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--num_step", type=int, default=256)
parser.add_argument("--shuffle_dataset", action='store_true') 
parser.add_argument("--save_vis", action='store_true')     
parser.add_argument('--only_test', action='store_true')
parser.add_argument('--only_train', action='store_true')  
parser.add_argument('--vis_process', action='store_true')   
parser.add_argument('--denoise', action='store_true')      
parser.add_argument('--extra_label', type=str, default="")                
args = parser.parse_args()

if args.dataset == "ucf101":
    num_classes = 24
    ds_path = f'{ds_root}/UCF101_24'
    if args.model == "v16l":
        from model_def.vgg16lstm import vgg16lstm as model
        from datasets.ucf101_24_dataset_vgg16lstm import UCF101_24_Dataset as dataset
        model_wgts_path = f"{proj_root}/model_param/ucf101_24_vgg16lstm_16_Full_LongRange.pt"
    elif args.model == "r2p1d":
        from model_def.r2plus1d import r2plus1d as model
        from datasets.ucf101_24_dataset_new import UCF101_24_Dataset as dataset
        model_wgts_path = f"{proj_root}/model_param/ucf101_24_r2p1d_16_Full_LongRange.pt"
    elif args.model == "r50l":
        from model_def.r50lstm import r50lstm as model
        from datasets.ucf101_24_dataset_new import UCF101_24_Dataset as dataset
        model_wgts_path = f"{proj_root}/model_param/ucf101_24_r50l_16_Full_LongRange.pt"
elif args.dataset == "epic":
    num_classes = 20
    ds_path = f'{ds_root}/epic'
    if args.model == "v16l":
        from model_def.vgg16lstm import vgg16lstm as model
        from datasets.epic_kitchens_dataset_vgg16lstm import EPIC_Kitchens_Dataset as dataset
        model_wgts_path = f"{proj_root}/model_param/epic_vgg16lstm_16_Full_LongRange.pt"
    elif args.model == "r2p1d":
        from model_def.r2plus1d import r2plus1d as model
        from datasets.epic_kitchens_dataset_new import EPIC_Kitchens_Dataset as dataset
        model_wgts_path = f"{proj_root}/model_param/epic_r2p1d_16_Full_LongRange.pt"
    elif args.model == "r50l":
        from model_def.r50lstm import r50lstm as model
        from datasets.epic_kitchens_dataset_new import EPIC_Kitchens_Dataset as dataset
        model_wgts_path = f"{proj_root}/model_param/epic_r50l_16_Full_LongRange.pt"

if args.save_vis:
    save_label = f"{args.dataset}_{args.model}_{args.vis_method}_{args.mode}_{args.keep_ratio}".replace(".", "_")
    if args.new_size != None:
        save_label = save_label + f"_{args.new_size}"
    vis_dir = f"{proj_root}/visual_res/auc_{save_label}"
    os.makedirs(vis_dir, exist_ok=True)

if args.vis_process:
    proc_vis_dir = f"{proj_root}/vis_cm/{args.dataset}_{args.model}_{args.mode}_{args.vis_method}"
    if args.extra_label != "":
        proc_vis_dir += f"{args.extra_label}"
    if args.new_size != None:
        proc_vis_dir += f"_{args.new_size}"
    if args.denoise:
        proc_vis_dir += f"_denoised"
    os.makedirs(proc_vis_dir, exist_ok=True)
else:
    proc_vis_dir = None

torch.manual_seed(2)
torch.cuda.manual_seed(2)

## ============== main ============== ##

phase_label = ""
if args.only_test:
    phase_label = "_test"
if args.only_train:
    phase_label = "_train"

res_buf = os.path.join(proj_root, 'exe_res', 
            f'{args.dataset}_{args.model}_{args.vis_method}_full{args.extra_label}{phase_label}.pt')
if args.vis_method == 'perturb':
    res_buf = res_buf.replace('.pt', '_summed.pt')
res_dic_lst = torch.load(res_buf)['val']
video_mask_dic = {res_dic["video_name"].split("/")[-1]: res_dic["mask"].astype(np.float32) for res_dic in res_dic_lst}
print(f"Loaded {res_buf}")

num_devices = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_tf = model(num_classes, with_softmax=True)
model_tf.load_weights(model_wgts_path)
model_tf.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_tf.to(device)
if args.multi_gpu:
    model_tf.parallel_model(device_ids=list(range(num_devices)))

video_dataset = dataset(ds_path, 16, 'long_range_last', 1, 6, False, bbox_gt=False, testlist_idx=1)

dataloader = DataLoader(video_dataset, batch_size=args.batch_size, shuffle=args.shuffle_dataset, num_workers=128)
cm_calculator = CausalMetric(model_tf, device)

if args.mode == "ins" or args.mode == "del":
    auc_sum = 0
else:
    ins_auc_sum = 0
    del_auc_sum = 0

for sample_idx, samples in enumerate(dataloader):
    clip_batch, class_ids, video_names, fidx_tensors = samples
    clip_batch = clip_batch.to(device).requires_grad_(False)

    mask_batch = [video_mask_dic[video_name.split("/")[-1]] for video_name in video_names]
    mask_batch = torch.from_numpy(np.stack(mask_batch, axis=0)).squeeze(1)  # bs x nt x 14 x 14
    if mask_batch.shape[-1] != clip_batch.shape[-1]:
        bs, _, nt, nrow, ncol = clip_batch.shape
        mask_batch = F.interpolate(mask_batch, size=(nrow, ncol), mode="bilinear")
    mask_batch = mask_batch.unsqueeze(1)    # bs x 1 x nt x 14 x 14
    
    if args.mode == "ins" or args.mode == "del":
        probs = cm_calculator.coarsly_evaluate(args.mode, clip_batch, mask_batch, class_ids, 
                    remove_method="fade", n_step=args.num_step, keep_topk=args.keep_ratio, new_size=args.new_size, 
                    visualize=args.vis_process, video_names=video_names, vis_dir=proc_vis_dir, denoise=args.denoise)
        for bidx, video_name in enumerate(video_names):
            scores = probs[bidx].numpy()
            auc = cal_auc(scores)
            video_name = video_name.split("/")[-1]
            print(f"{video_name}: {auc}")
            if args.save_vis:
                plot_causal_metric_curve(scores, show_txt = f"{video_name}_{args.mode}, auc = {auc:.2f}",
                                            save_dir = join(vis_dir, f"{video_name}_{args.mode}.png"))
            auc_sum += auc
    else:
        ins_probs = cm_calculator.coarsly_evaluate("ins", clip_batch, mask_batch, class_ids, 
                    remove_method="fade", n_step=args.num_step, keep_topk=args.keep_ratio, new_size=args.new_size, 
                    visualize=args.vis_process, video_names=video_names, vis_dir=proc_vis_dir, denoise=args.denoise)
        del_probs = cm_calculator.coarsly_evaluate("del", clip_batch, mask_batch, class_ids, 
                    remove_method="fade", n_step=args.num_step, keep_topk=args.keep_ratio, new_size=args.new_size, 
                    visualize=args.vis_process, video_names=video_names, vis_dir=proc_vis_dir, denoise=args.denoise)
        for bidx, video_name in enumerate(video_names):
            del_scores = del_probs[bidx].numpy()
            ins_scores = ins_probs[bidx].numpy()
            del_auc = cal_auc(del_scores)
            ins_auc = cal_auc(ins_scores)
            video_name = video_name.split("/")[-1]
            print(f"{video_name}: del:{del_auc}/ins:{ins_auc}")
            if args.save_vis:
                plot_causal_metric_curve(del_scores, show_txt = f"{video_name}_del, auc = {del_auc:.2f}",
                                            save_dir = join(vis_dir, f"{video_name}_del.png"))
                plot_causal_metric_curve(ins_scores, show_txt = f"{video_name}_ins, auc = {ins_auc:.2f}",
                                            save_dir = join(vis_dir, f"{video_name}_ins.png"))
            del_auc_sum += del_auc
            ins_auc_sum += ins_auc

if args.mode == "ins" or args.mode == "del":
    print(f"Average: {auc_sum/len(res_dic_lst)}")
else:
    print(f"Average: {del_auc_sum/len(res_dic_lst)}/{ins_auc_sum/len(res_dic_lst)}")
