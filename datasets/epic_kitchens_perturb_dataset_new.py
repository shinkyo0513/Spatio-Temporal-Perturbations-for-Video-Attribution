import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

import os
from os.path import join, isdir, isfile
from os import listdir

import pickle
# from PIL import Image
from glob import glob
import bisect
import numpy as np
import math
import random
import pandas as pd
import ast
from skimage.filters import gaussian

import accimage
torchvision.set_image_backend('accimage')
from accimage import Image

import sys
sys.path.append("..")
from utils.GaussianSmoothing import GaussianSmoothing
from utils import LongRangeSample 
from utils import ImageShow

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

crt_dir = os.path.dirname(os.path.realpath(__file__))

import copy
class SampledVideoClips (object):
    def __init__ (self, video_names, root_path, clip_length_in_frames=16, 
                    frame_rate=1, clips_per_video=1, sample_mode='random'):
        self.video_names = video_names
        self.root_path = root_path
        self.clip_length_in_frames = clip_length_in_frames
        self.frame_rate = frame_rate
        self.clips_per_video = clips_per_video
        self.sample_mode = sample_mode

        self.video_numf_dict = {}
        self.video_stf_dict = {}
        for video_name in video_names:
            video_path = join(root_path, video_name)
            frame_lst = sorted(listdir(video_path))
            num_frame = len(frame_lst)
            start_frame = int(frame_lst[0][-14:-4])
            self.video_numf_dict[video_name] = num_frame
            self.video_stf_dict[video_name] = start_frame

        self.compute_clips()

    def compute_clips_for_video (self, video_name):
        numf = self.video_numf_dict[video_name]
        fr = self.frame_rate
        cl = self.clip_length_in_frames
        cn = self.clips_per_video

        video_clips = []
        if 'long_range' in self.sample_mode:
            # For long range sampling
            if self.sample_mode == 'long_range_random':
                for ci in range(cn):
                    frame_indexes_in_clip = LongRangeSample.long_range_rand(numf, cl)
                    video_clips.append(frame_indexes_in_clip)
            elif self.sample_mode == 'long_range_first':
                for ci in range(cn):
                    frame_indexes_in_clip = LongRangeSample.long_range_first(numf, cl)
                    video_clips.append(frame_indexes_in_clip)
            elif self.sample_mode == 'long_range_last':
                for ci in range(cn):
                    frame_indexes_in_clip = LongRangeSample.long_range_last(numf, cl)
                    video_clips.append(frame_indexes_in_clip)
        else:
            max_num_clips = math.floor(numf/fr)-cl+1
            assert max_num_clips >= cn, f"For video: {video_name}, "+\
                    f"the length of {numf} isn't enough for sampling "+\
                    f"{cn} clips with frame rate of {fr}."
            # For continuous sampling
            if self.sample_mode == 'random':
                clip_indexes = random.sample(range(max_num_clips), cn)
            elif self.sample_mode == 'fixed':
                # clip_indexes = list(range(0, max_num_clips, max_num_clips//cn))
                clip_indexes = [max_num_clips*(2*idx+1)//(2*cn) for idx in range(cn)]
            else:
                raise Exception(f"sample_mode do not support, given {self.sample_mode}")
            for clip_idx in clip_indexes:
                first_frame = clip_idx*fr
                frame_indexes_in_clip = [first_frame+offset*fr for offset in range(cl)]
                video_clips.append(frame_indexes_in_clip)

        return video_clips

    def compute_clips (self):
        self.video_clips_dict = {}
        for video_name in self.video_names:
            video_clips = self.compute_clips_for_video(video_name)
            self.video_clips_dict[video_name] = video_clips

    def num_clips (self):
        return len(self.video_names)*self.clips_per_video

    def get_clip_location (self, idx):
        video_idx = idx//self.clips_per_video
        clip_idx = idx%self.clips_per_video
        return video_idx, clip_idx

    def get_clip (self, idx):
        if idx >= self.num_clips():
            raise IndexError("Index {} out of range "
                             "({} number of clips)".format(idx, self.num_clips()))
        video_idx, clip_idx = self.get_clip_location(idx)
        video_name = self.video_names[video_idx]
        video_stf = self.video_stf_dict[video_name]
        clip_frame_indices = self.video_clips_dict[video_name][clip_idx]
        clip_frames = [Image(join(self.root_path, video_name, 
                        f"frame_{frame_idx+video_stf:010d}.jpg")) for frame_idx in clip_frame_indices]

        return clip_frames, video_idx, clip_frame_indices

def perturb_frames (frames_tensor, heatmaps_tensor, perturb_ratio, perturb_mode='remove', fade_type='zero'):
    # frames: 3x16x112x112; heatmaps: 1x16x h x w
    fch, fnt, fnrow, fncol = frames_tensor.shape
    hch, hnt, hnrow, hncol = heatmaps_tensor.shape
    assert fnt == hnt
    # if hnrow!=fnrow or hncol!=fncol:
    #     heatmaps_tensor = F.interpolate(heatmaps_tensor, (fnrow, fncol), 
    #                         mode='bilinear', align_corners=False)    # 1x16x112x112

    if perturb_mode == 'remove':
        perturb_k = int(perturb_ratio * fnt * fnrow * fncol)
        _, k_idxs = heatmaps_tensor.reshape(1, -1).topk(perturb_k, dim=1, largest=True)
    elif perturb_mode == 'keep':
        perturb_k = int((1-perturb_ratio) * fnt * fnrow * fncol)
        _, k_idxs = heatmaps_tensor.reshape(1, -1).topk(perturb_k, dim=1, largest=False)
    k_idxs = k_idxs.repeat_interleave(fch, dim=0)
    perturbed_frames_tensor = frames_tensor.clone()
    perturbed_frames_tensor.view(fch, -1).scatter_(1, k_idxs, 0) # Set the topk elements in frames_tensor to be 0
    return perturbed_frames_tensor, heatmaps_tensor

def perturb_frames_by_block (frames_tensor, heatmaps_tensor, perturb_ratio, perturb_mode='remove', block_size=7):
    # frames: 3x16x112x112; heatmaps: 1x16x h x w
    fch, fnt, fnrow, fncol = frames_tensor.shape
    hch, hnt, hnrow, hncol = heatmaps_tensor.shape
    assert hnrow%block_size==0 and hncol%block_size==0
    new_heatmap_size = hnrow / block_size
    device = frames_tensor.device

    k = torch.ones((1, 1, block_size, block_size)) / (block_size*block_size)
    new_heatmaps_tensor = heatmaps_tensor.transpose(0, 1)  # 16 x 1x h x w
    new_heatmaps_tensor = F.conv2d(new_heatmaps_tensor, k, stride=block_size, padding=0)    # 16 x 1 x h' x w'
    new_heatmaps_tensor = new_heatmaps_tensor.transpose(0, 1)    # 1 x 16 x h' x w'

    if perturb_mode == 'remove':
        sal_order = new_heatmaps_tensor.reshape(1, -1).argsort(dim=-1, descending=True) # 1 x 16*h'*w'
        remove_topk = int(perturb_ratio * fnt * new_heatmap_size * new_heatmap_size)
    elif perturb_mode == 'keep':
        sal_order = new_heatmaps_tensor.reshape(1, -1).argsort(dim=-1, descending=False) # 1 x 16*h'*w'
        remove_topk = int((1-perturb_ratio) * fnt * new_heatmap_size * new_heatmap_size)
    remove_pos = sal_order[:, :remove_topk]
    idx = torch.arange(hch, dtype=torch.long).view(-1,1).to(device)

    remove_masks = torch.ones_like(new_heatmaps_tensor).to(device)  # 1: keep; 0: remove
    remove_masks.reshape(1, -1)[idx, remove_pos] = 0
    remove_masks = F.interpolate(remove_masks, size=(hnrow, hncol), mode='nearest')  # 16 x 1 x h x w
    perturbed_frames_tensor = frames_tensor * remove_masks
    return perturbed_frames_tensor, new_heatmaps_tensor

class EPIC_Kitchens_Dataset (Dataset):
    def __init__ (self, root_dir, frames_per_clip, sample_mode, 
                    num_clips, heatmap_dir, perturb_ratio, perturb_mode='remove',
                    frame_rate=1, train=True, testlist_idx=2, 
                    smoothed_perturb=False, smooth_sigma=10, 
                    perturb_by_block=False, block_size=7,
                    noised=False):
        self.root_dir = root_dir
        self.rgb_file = join(root_dir, "seg_train")
        if train:
            self.annot_file = crt_dir.replace("datasets", "my_epic_annot/Valid_seg_top20_train.csv")
        else:
            if testlist_idx == 1:
                testlist = "top20_500"
            elif testlist_idx == 2:
                testlist = "top20_100"
            self.annot_file = crt_dir.replace("datasets", f"my_epic_annot/Valid_seg_{testlist}_val.csv")
        # self.perturb = perturb

        self.frames_per_clip = frames_per_clip
        self.frame_rate = frame_rate
        self.sample_mode = sample_mode
        self.num_clips = num_clips
        # self.fade_type = fade_type
        # self.bbox_gt = bbox_gt

        seg_info_df = pd.read_csv(self.annot_file)
        self.seg_annot_dict = {}
        for idx, seg_info in seg_info_df.iterrows():
            v_id = seg_info["video_id"]
            seg_id = seg_info["seg_id"]
            seg_noun = seg_info["noun"]
            seg_verb = seg_info["verb"]
            seg_name = join(f"{v_id[:3]}", f"{v_id}", f"{v_id}_{seg_id}-{seg_verb}-{seg_noun}")
            self.seg_annot_dict[seg_name] = seg_info

        all_list = list(self.seg_annot_dict.keys())
        self.sltd_list = sorted(all_list)
        
        self.pre_trans = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor()
        ])

        self.norm_trans = transforms.Compose([
            transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
        ])

        self.all_clips = SampledVideoClips(self.sltd_list, self.rgb_file, frames_per_clip, 
                                                frame_rate, num_clips, sample_mode)

        self.heatmap_dir = heatmap_dir
        self.perturb_ratio = perturb_ratio
        self.perturb_mode = perturb_mode
        self.smoothed_perturb = smoothed_perturb
        self.smooth_sigma = smooth_sigma
        self.perturb_by_block = perturb_by_block
        self.block_size = block_size
        self.noised = noised
        
        if heatmap_dir != 'random':
            self.heatmap_dict = torch.load(heatmap_dir)
            self.heatmap_dict = self.heatmap_dict['train'] if train else self.heatmap_dict['val']
            self.heatmap_dict = {res_dic['video_name']: res_dic['mask'] for res_dic in self.heatmap_dict}
            # print(self.heatmap_dict.keys())

    def __len__ (self):
        return self.all_clips.num_clips()

    def __getitem__ (self, idx):
        # clip_frames: frame images in the clip
        # video_idx: the index of video which the clip belonging to
        # clip_fidx: indices of frames in the clip (relative to one segment, i.e., 0~seg_len-1)
        clip_frames, seg_idx, clip_fidx = self.all_clips.get_clip(idx)
        seg_name = self.sltd_list[seg_idx]
        label = int(self.seg_annot_dict[seg_name]["noun_label"])

        clip_tensor = torch.stack([self.transform(clip_frame) for clip_frame in 
                                    clip_frames], dim=1)    # 3x16x112x112

        if self.heatmap_dir != 'random':
            heatmaps_np = self.heatmap_dict[video_name_regu].astype(np.float)
            heatmaps_tensor = torch.from_numpy(heatmaps_np).to(dtype=torch.float)   # 1x16x h x w
            if heatmaps_tensor.shape[2] != clip_tensor.shape[2]:   # For Grad-CAM
                heatmaps_tensor = F.interpolate(heatmaps_tensor, (112, 112), 
                                    mode='bilinear', align_corners=False)    # 1x16x112x112
        else:
            heatmaps_tensor = torch.randn((1,16,112,112))

        if self.noised:
            noise = torch.randn((1,16,112,112)) * heatmaps_tensor.max() * 0.2
            heatmaps_tensor += noise

        if self.smoothed_perturb:
            smoothed_heatmaps = [gaussian(heatmaps_tensor[0,fidx,...].numpy(), 
                                        sigma=self.smooth_sigma) for fidx in range(16)]
            heatmaps_tensor = torch.from_numpy(np.stack(smoothed_heatmaps, axis=0)).unsqueeze(0)     # 1x16x112x112

        if self.perturb_by_block:
            perturbed_clip_tensor, heatmaps_tensor = perturb_frames_by_block(clip_tensor, heatmaps_tensor, 
                                                        self.perturb_ratio, self.perturb_mode, self.block_size)
        else:
            perturbed_clip_tensor, heatmaps_tensor = perturb_frames(clip_tensor, heatmaps_tensor, 
                                                        self.perturb_ratio, self.perturb_mode)

        perturbed_clip_tensor = torch.stack([self.norm_trans(f.squeeze(1)) for f in 
                                    perturbed_clip_tensor.split(1, dim=1)], dim=1)    # 3x16x112x112
        clip_fidx_tensor = torch.tensor(clip_fidx).long()   # clip_len
        return perturbed_clip_tensor, label, seg_name, clip_fidx_tensor, heatmaps_tensor
