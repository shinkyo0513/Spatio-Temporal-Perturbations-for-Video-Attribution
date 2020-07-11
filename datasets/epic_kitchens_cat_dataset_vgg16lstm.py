import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
from os.path import join, isdir, isfile
from os import listdir

import pickle
from PIL import Image
from glob import glob
import bisect
import numpy as np
import math
import random
import pandas as pd
import ast

import sys
sys.path.append("..")
from utils.GaussianSmoothing import GaussianSmoothing
from utils import LongRangeSample 

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
        clip_frames = [copy.deepcopy(Image.open(join(self.root_path, video_name, 
                        f"frame_{frame_idx+video_stf:010d}.jpg"))) for frame_idx in clip_frame_indices]
        return clip_frames, video_idx, clip_frame_indices

class EPIC_Kitchens_Dataset (Dataset):
    def __init__ (self, root_dir, frames_per_clip, sample_mode, 
                    num_clips, frame_rate=1, train=True, 
                    bbox_gt = False, testlist_idx=2, 
                    perturb=None, fade_type='gaussian'):
        self.root_dir = root_dir
        self.rgb_file = join(root_dir, "frames_rgb_flow/rgb/cat_obj_segs")
        # self.rgb_file = join(root_dir, "frames_rgb_flow/rgb/train")
        if train:
            self.annot_file = crt_dir.replace("datasets", "my_epic_annot/Valid_seg_top20_train.csv")
        else:
            if testlist_idx == 1:
                testlist = "top20_500"
            elif testlist_idx == 2:
                testlist = "top20_100"
            self.annot_file = crt_dir.replace("datasets", f"my_epic_annot/Valid_seg_{testlist}_val_cat.csv")
        self.perturb = perturb

        self.frames_per_clip = frames_per_clip
        self.frame_rate = frame_rate
        self.sample_mode = sample_mode
        self.num_clips = num_clips
        self.fade_type = fade_type
        self.bbox_gt = bbox_gt

        seg_info_df = pd.read_csv(self.annot_file)
        self.seg_annot_dict = {}
        for idx, seg_info in seg_info_df.iterrows():
            v_id = seg_info["video_id"]
            seg_id = seg_info["seg_id"]
            seg_noun = seg_info["noun"]
            seg_verb = seg_info["verb"]
            # seg_name = join(f"{v_id[:3]}", f"{v_id}", f"{v_id}_{seg_id}-{seg_verb}-{seg_noun}")
            seg_name = join(f"{v_id}_{seg_id}-{seg_verb}-{seg_noun}")
            self.seg_annot_dict[seg_name] = seg_info

        all_list = list(self.seg_annot_dict.keys())
        self.sltd_list = sorted(all_list)
        
        self.transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.50463, 0.45796, 0.4076], [1.0, 1.0, 1.0]),
            # transforms.RandomErasing(p=1.0, scale=(0.8, 0.8), ratio=(1.0, 1.0), value=0),
        ])

        self.all_clips = SampledVideoClips(self.sltd_list, self.rgb_file, frames_per_clip, 
                                                frame_rate, num_clips, sample_mode)

        if bbox_gt:
            self.seg_bbox_dict = self.set_seg_bbox_dict()

        if perturb:
            self.seg_grounds_dict = self.set_seg_grounds_dict()
            # ave_ratio = self.cal_ground_area()
            # print(f'The aberage ratio of grounds size in frame {ave_ratio}.')

    def __len__ (self):
        return self.all_clips.num_clips()

    def __getitem__ (self, idx):
        # clip_frames: frame images in the clip
        # video_idx: the index of video which the clip belonging to
        # clip_fidx: indices of frames in the clip (relative to one segment, i.e., 0~seg_len-1)
        clip_frames, seg_idx, clip_fidx = self.all_clips.get_clip(idx)
        seg_name = self.sltd_list[seg_idx]
        label = int(self.seg_annot_dict[seg_name]["noun_label"])

        seg_info = self.seg_annot_dict[seg_name]
        st = seg_info["start_frame"]
        mid = seg_info["middle_frame"]
        ground = seg_info["ground"]
        if ground == 'left':
            temporal_ground_mask = [1 if fidx + st < mid else 0 for fidx in clip_fidx]
        elif ground == 'right':
            temporal_ground_mask = [0 if fidx + st < mid else 1 for fidx in clip_fidx]
        temporal_ground_mask = torch.tensor(temporal_ground_mask)

        clip_tensor = torch.stack([self.transform(clip_frame) for clip_frame in 
                                    clip_frames], dim=1)    # 3x16x112x112
        clip_fidx_tensor = torch.tensor(clip_fidx).long()   # clip_len

        if self.bbox_gt:
            clip_bbox_tensor = self.get_clip_bbox_tensor(seg_name, clip_fidx, srch_hw=3)
            return clip_tensor, label, seg_name, clip_fidx_tensor, clip_bbox_tensor
        
        if self.perturb:
            clip_grounds = self.get_clip_grounds(seg_name, clip_fidx)
            clip_tensor = self.perturb_clip_tensor(clip_tensor, clip_grounds, 
                                                    self.perturb, fade_type=self.fade_type)
            return clip_tensor, label, seg_name, clip_grounds

        return clip_tensor, label, seg_name, clip_fidx_tensor, temporal_ground_mask

    def set_seg_bbox_dict (self):
        seg_bbox_dict = {}
        delta_x = (320 - 224) / 2
        delta_y = (240 - 224) / 2
        max_x = max_y = 223
        for seg_name in self.sltd_list:
            seg_info = self.seg_annot_dict[seg_name]
            seg_sf = int(seg_info["start_frame"])
            seg_ef = int(seg_info["stop_frame"])
            seg_bbox = ast.literal_eval(seg_info["bounding_boxes"])
            seg_numf = seg_ef - seg_sf + 1

            seg_bbox_lst = [(),]*seg_numf
            for fidx_wbbox, bbox in seg_bbox.items():
                y, x, dy, dx = bbox
                y = int(240*y / 1080)
                dy = int(240*dy / 1080)
                x = int(320*x / 1920)
                dx = int(320*dx / 1920)

                x0 = min(max(x-delta_x, 0), max_x)
                y0 = min(max(y-delta_y, 0), max_y)
                x1 = min(max(x+dx-delta_x, 0), max_x)
                y1 = min(max(y+dy-delta_y, 0), max_y)
                seg_bbox_lst[fidx_wbbox-seg_sf] = (x0, y0, x1, y1)
            
            seg_bbox_dict[seg_name] = seg_bbox_lst
        return seg_bbox_dict

    def get_seg_bbox_dict (self):
        if self.bbox_gt:
            return self.seg_bbox_dict
    
    # srch_hw: search half width
    def get_clip_bbox_tensor (self, seg_name, clip_fidx, srch_hw=6):
        seg_bbox_lst = self.seg_bbox_dict[seg_name]

        clip_bbox_lst = [(0,0,0,0),]*len(clip_fidx)
        for fidx, bbox in enumerate(seg_bbox_lst):
            if bbox != ():
                for srch_fidx in range(fidx-srch_hw, fidx+srch_hw+1):
                    if srch_fidx in clip_fidx:
                        idx_in_clip = clip_fidx.index(srch_fidx)
                        clip_bbox_lst[idx_in_clip] = bbox

        clip_bbox_tensor = torch.tensor(clip_bbox_lst).long()   # 16 x 4
        return clip_bbox_tensor
