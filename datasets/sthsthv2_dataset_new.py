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
from PIL import Image
import glob
import bisect
import numpy as np
import math
import random
import json

import accimage
torchvision.set_image_backend('accimage')
from accimage import Image

import sys
sys.path.append(".")
sys.path.append("..")

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

from utils.GaussianSmoothing import GaussianSmoothing
from utils import LongRangeSample 

import copy
class SampledVideoClips (object):
    def __init__ (self, video_samples, root_path, clip_length_in_frames=16, 
                    frame_rate=1, clips_per_video=1, sample_mode='random'):
        self.video_samples = video_samples
        self.root_path = root_path
        self.clip_length_in_frames = clip_length_in_frames
        self.frame_rate = frame_rate
        self.clips_per_video = clips_per_video
        self.sample_mode = sample_mode

        self.video_ids = []
        self.video_numf_dict = {}
        for sample in video_samples:
            video_id = sample['id']
            action_label = sample['template']
            frame_path = join(root_path, video_id)
            num_frame = len(glob.glob(os.path.join(frame_path, '*.jpg')))

            self.video_ids.append(video_id)
            self.video_numf_dict[video_id] = num_frame

        self.compute_clips()

    def compute_clips_for_video (self, video_id):
        numf = self.video_numf_dict[video_id]
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
        for video_id in self.video_ids:
            video_clips = self.compute_clips_for_video(video_id)
            self.video_clips_dict[video_id] = video_clips

    def num_clips (self):
        return len(self.video_ids)*self.clips_per_video

    def get_clip_location (self, idx):
        video_idx = idx//self.clips_per_video
        clip_idx = idx%self.clips_per_video
        return video_idx, clip_idx

    def get_clip (self, idx):
        if idx >= self.num_clips():
            raise IndexError("Index {} out of range "
                             "({} number of clips)".format(idx, self.num_clips()))
        video_idx, clip_idx = self.get_clip_location(idx)
        video_id = self.video_ids[video_idx]
        clip_frame_indices = self.video_clips_dict[video_id][clip_idx]
        clip_frames = [Image(join(self.root_path, video_id, 
                        format(frame_idx+1, '06d')+'.jpg')) for frame_idx in clip_frame_indices]

        return clip_frames, video_idx, clip_frame_indices

class SthSthV2_Dataset (Dataset):
    def __init__ (self, root_dir, frames_per_clip, sample_mode, 
                  num_clips, frame_rate=1, train=True, 
                  testlist_idx=2, labels_set='full', train_set='7000'):
        self.root_dir = root_dir
        self.rgb_dir = join(root_dir, '20bn-something-something-v2-frames')
        
        self.train = train
        if self.train:
            if train_set == '7000':
                self.annot_file = join(proj_root, 'my_sthsthv2_annot', 'train_7000.json')
            elif train_set == '10000':
                self.annot_file = join(proj_root, 'my_sthsthv2_annot', 'train_10000.json')
        else:
            if testlist_idx == 1:
                self.annot_file = join(proj_root, 'my_sthsthv2_annot', 'val_1000.json')
            elif testlist_idx == 2:
                self.annot_file = join(proj_root, 'my_sthsthv2_annot', 'val_500.json')
        
        assert labels_set in ['full', 'top25']
        self.labels_set = labels_set
        if self.labels_set == 'full':
            labels_index_path = join(ds_root, 'something_something_v2/annotations', 'something-something-v2-labels.json')
        elif self.labels_set == 'top25':
            labels_index_path = join(proj_root, 'my_sthsthv2_annot', 'top25_labels_index.json')
        with open(labels_index_path) as f:
            self.labels_index_dict = json.load(f)

        self.frames_per_clip = frames_per_clip
        self.frame_rate = frame_rate
        self.sample_mode = sample_mode
        self.num_clips = num_clips

        with open(self.annot_file) as f:
            self.sample_list = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
        ])
        # self.transform = transforms.Compose([
        #     transforms.CenterCrop((224,224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])

        self.video_clips = SampledVideoClips(self.sample_list, self.rgb_dir, frames_per_clip, 
                                                frame_rate, num_clips, sample_mode)

    def __len__ (self):
        return self.video_clips.num_clips()

    def __getitem__ (self, idx):
        clip_frames, video_idx, clip_frame_indices = self.video_clips.get_clip(idx)
        sample = self.sample_list[video_idx]
        # label = self.video_annot_dict[video_name]['label']
        video_id = sample['id']
        action_label = sample['template'].replace('[', '').replace(']', '')
        label_index = int(self.labels_index_dict[action_label])
        # print(action_label, '-->', label_index)

        clip_tensor = torch.stack([self.transform(clip_frame) for clip_frame in 
                                    clip_frames], dim=1)    # 3x16x112x112
        clip_fidx_tensor = torch.tensor(clip_frame_indices).long()   # clip_len

        # video_name = '/'.join((action_label, str(video_id)))
        # print(label_index, action_label)
        # print(video_id)
        return clip_tensor, label_index, video_id, clip_fidx_tensor, action_label
