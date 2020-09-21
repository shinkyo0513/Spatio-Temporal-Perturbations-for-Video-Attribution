import os
from os.path import join, isdir, isfile

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

from utils.Faithfulness import *
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
# parser.add_argument("--res_buf", type=str)
parser.add_argument("--multi_gpu", action='store_true')
# parser.add_argument("--keep_ratio", type=float, default=1.0)
# parser.add_argument("--new_size", type=int, default=None)
# parser.add_argument("--save_vis", action='store_true')     
parser.add_argument('--only_test', action='store_true')
parser.add_argument('--only_train', action='store_true')       
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

# if args.save_vis:
#     save_label = f"{args.dataset}_{args.model}_{args.vis_method}".replace(".", "_")
#     # if args.new_size != None:
#     #     save_label = save_label + f"_{args.new_size}"
#     vis_dir = f"{proj_root}/visual_res/auc_{save_label}"
#     os.makedirs(vis_dir, exist_ok=True)
# else:
#     vis_dir = None
#     print(f"The visualization results will not be visualized and save.")

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

dataloader = DataLoader(video_dataset, batch_size=20, shuffle=False, num_workers=128)
f_calculator = Faithfulness(model_tf, device)

pearson_dict = {}
avg_pearson = 0.0
for sample_idx, samples in enumerate(dataloader):
    clip_batch, class_ids, video_names, fidx_tensors = samples

    mask_batch = [video_mask_dic[video_name.split("/")[-1]] for video_name in video_names]
    mask_batch = torch.from_numpy(np.stack(mask_batch, axis=0)).squeeze(1)  # bs x nt x h x w
    if mask_batch.shape[-1] != clip_batch.shape[-1]:
        bs, _, nt, nrow, ncol = clip_batch.shape
        mask_batch = F.interpolate(mask_batch, size=(nrow, ncol), mode="bilinear")
    mask_batch = mask_batch.unsqueeze(1)    # bs x 1 x nt x h x w

    clip_batch = clip_batch.to(device).requires_grad_(False)
    
    pearson_coeffs = f_calculator.calculate(clip_batch, mask_batch, class_ids, remove_method="fade")

    for bidx, video_name in enumerate(video_names):
        p = pearson_coeffs[bidx].item()
        if not np.isnan(p):
            pearson_dict[video_name] = p
            avg_pearson += p
            print(f'{video_name}: {p:.5f}')

avg_pearson = sum(pearson_dict.values()) / len(pearson_dict.keys())
# avg_pearson = avg_pearson / len(video_dataset)
print(f'Avg: {avg_pearson:.5f}')