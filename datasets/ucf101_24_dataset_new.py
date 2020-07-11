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

import accimage
torchvision.set_image_backend('accimage')
from accimage import Image

import sys
sys.path.append("..")
from utils.GaussianSmoothing import GaussianSmoothing
from utils import LongRangeSample 

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
        for video_name in video_names:
            video_path = join(root_path, video_name)
            frame_names = [frame_name for frame_name in listdir(video_path) if "jpg" in frame_name]
            num_frame = len(frame_names)
            # num_frame = len(listdir(video_path))
            self.video_numf_dict[video_name] = num_frame

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
        clip_frame_indices = self.video_clips_dict[video_name][clip_idx]
        clip_frames = [Image(join(self.root_path, video_name, 
                        format(frame_idx+1, '05d')+'.jpg')) for frame_idx in clip_frame_indices]

        return clip_frames, video_idx, clip_frame_indices

class UCF101_24_Dataset (Dataset):
    def __init__ (self, root_dir, frames_per_clip, sample_mode, 
                    num_clips, frame_rate=1, train=True, 
                    bbox_gt=False, testlist_idx=2,
                    perturb=None, fade_type='gaussian'):
        self.root_dir = root_dir
        self.annot_file = join(root_dir, 'annotations/pyannot.pkl')
        # Change split_file for smaller test set
        if testlist_idx == 1:
            self.split_file = join(root_dir, 'splits/testlist01.txt')
        elif testlist_idx == 2:
            self.split_file = join(root_dir, 'splits/testlist02.txt')
        self.rgb_file = join(root_dir, 'images')
        self.train = train
        self.perturb = perturb
        self.bbox_gt = bbox_gt

        self.frames_per_clip = frames_per_clip
        self.frame_rate = frame_rate
        self.sample_mode = sample_mode
        self.num_clips = num_clips
        self.fade_type = fade_type

        with open(self.annot_file, 'rb') as f:
            # video_annot_dict[video_name]: (['annotations', 'numf', 'label'])
            self.video_annot_dict = pickle.load(f)

        all_list = list(self.video_annot_dict.keys())
        test_list = open(self.split_file, 'r').read().splitlines()
        self.sltd_list = [video_name for video_name in all_list 
                            if video_name not in test_list] if train else test_list
        self.sltd_list = sorted(self.sltd_list)
        
        self.transform = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
            # transforms.RandomErasing(p=1.0, scale=(0.8, 0.8), ratio=(1.0, 1.0), value=0),
        ])

        self.video_clips = SampledVideoClips(self.sltd_list, self.rgb_file, frames_per_clip, 
                                                frame_rate, num_clips, sample_mode)

        self.video_grounds_dict = self.set_video_grounds_dict()
        # if perturb:
        #     self.video_grounds_dict = self.set_video_grounds_dict()
            # ave_ratio = self.cal_ground_area()
            # print(f'The aberage ratio of grounds size in frame {ave_ratio}.')

    def __len__ (self):
        return self.video_clips.num_clips()

    def __getitem__ (self, idx):
        clip_frames, video_idx, clip_frame_indices = self.video_clips.get_clip(idx)
        video_name = self.sltd_list[video_idx]
        label = self.video_annot_dict[video_name]['label']

        clip_tensor = torch.stack([self.transform(clip_frame) for clip_frame in 
                                    clip_frames], dim=1)    # 3x16x112x112
        clip_fidx_tensor = torch.tensor(clip_frame_indices).long()   # clip_len

        if self.bbox_gt:
            clip_grounds = self.get_clip_grounds(video_idx, clip_frame_indices)
            clip_grounds_tensor = torch.tensor(clip_grounds).long()
            return clip_tensor, label, video_name, clip_fidx_tensor, clip_grounds_tensor
        
        if self.perturb:
            clip_grounds = self.get_clip_grounds(video_idx, clip_frame_indices)
            clip_grounds_tensor = torch.tensor(clip_grounds).long()
            clip_tensor = self.perturb_clip_tensor(clip_tensor, clip_grounds, 
                                                    self.perturb, fade_type='gaussian')
            return clip_tensor, label, video_name, clip_fidx_tensor, clip_grounds_tensor

        return clip_tensor, label, video_name, clip_fidx_tensor

    def set_video_grounds_dict (self):
        video_grounds_dict = {}
        for video_name in self.sltd_list:
            video_annots = self.video_annot_dict[video_name]['annotations']
            video_numf = self.video_annot_dict[video_name]['numf']
            video_label = self.video_annot_dict[video_name]['label']

            video_grounds = [(0,0,0,0)]*video_numf
            for video_annot in video_annots:
                sf = video_annot['sf']
                ef = video_annot['ef']
                if video_annot['label'] == video_label:
                    for f_idx in range(sf, ef):
                        # video_grounds[f_idx] = video_annot['boxes'][f_idx-sf].tolist()
                        x, y, dx, dy = list(video_annot['boxes'][f_idx-sf].astype(np.int16))
                        x0 = min(math.floor(112*x / 320), 111)
                        y0 = min(math.floor(112*y / 240), 111)
                        x1 = min(math.ceil(112*(x+dx) / 320), 111)
                        y1 = min(math.ceil(112*(y+dy) / 240), 111)
                        video_grounds[f_idx] = (x0, y0, x1, y1)

            video_grounds_dict[video_name] = video_grounds
        return video_grounds_dict
            
    def get_clip_grounds (self, video_idx, clip_frame_indices):
        video_name = self.sltd_list[video_idx]
        video_grounds = self.video_grounds_dict[video_name]
        clip_grounds = []
        for f_idx in clip_frame_indices:
            clip_grounds.append( video_grounds[f_idx] )
        return clip_grounds

    # TODO: perturb frames by grounds need to be change
    def perturb_clip_tensor (self, clip_tensor, clip_grounds, mode, fade_type='gaussian'):

        # Using self-implemented gaussian perturbation
        def get_perturb_masks(clip_grounds):
            pmasks = []
            for f_idx, f_grounds in enumerate(clip_grounds):
                f_pmask = torch.zeros(1, 1, 112, 112)
                if f_grounds != (0,0,0,0):
                    for ground in f_grounds:
                        x, y, dx, dy = list(ground.astype(np.int16))
                        x = math.floor(112*x / 320)
                        y = math.floor(112*y / 240)
                        dx = math.ceil(112*dx / 320)
                        dy = math.ceil(112*dy / 240)
                        f_pmask[:, :, y:y+dy, x:x+dx] = 1.0
                pmasks.append(f_pmask)
            pmasks_tensor = torch.cat(pmasks, dim=1)
            return pmasks_tensor

        pmasks_tensor = get_perturb_masks(clip_grounds) # 1x16x112x112
        if fade_type == 'gaussian':
            smoothing = GaussianSmoothing(channels=3, kernel_size=15, sigma=21)

        p_clip_tensor = []
        for f_idx in range(clip_tensor.shape[1]):
            f_tensor = clip_tensor[:,f_idx,:,:].unsqueeze(0)    #1x3x112x112
            if fade_type == 'gaussian':
                blurred_f_tensor = smoothing(F.pad(f_tensor, (7, 7, 7, 7)))
            elif fade_type == 'black':
                blurred_f_tensor = torch.zeros_like(f_tensor)
            elif fade_type == 'random':
                blurred_f_tensor = torch.stack([
                    torch.randn((112,112))*0.22803+0.43216,
                    torch.randn((112,112))*0.22145+0.394666,
                    torch.randn((112,112))*0.216989+0.37645,
                ], dim=0).unsqueeze(0)
            elif fade_type == 'mean':
                blurred_f_tensor = torch.stack([
                    torch.zeros((112,112))+0.43216,
                    torch.zeros((112,112))+0.394666,
                    torch.zeros((112,112))+0.37645,
                ], dim=0).unsqueeze(0)
            else:
                raise Exception('No fade_type called '+fade_type)

            f_pmask = pmasks_tensor[:,f_idx,:,:].unsqueeze(0)   #1x1x112x112
            if mode=='delete':   # delete annotation regions
                p_f_tensor = f_pmask * blurred_f_tensor + (1.0-f_pmask) * f_tensor
            elif mode=='preserve':
                p_f_tensor = (1.0-f_pmask) * blurred_f_tensor + f_pmask * f_tensor
            elif mode=='full':
                p_f_tensor = blurred_f_tensor
            p_clip_tensor.append(p_f_tensor.squeeze(0)) #3x112x112
        p_clip_tensor = torch.stack(p_clip_tensor, dim=1)   #3x16x112x112
        return p_clip_tensor

    def cal_ground_area (self):
        area_sum = 0
        num_ground = 0
        for video_name, video_grounds in self.video_grounds_dict.items():
            for f_idx, f_grounds in enumerate(video_grounds):
                if f_grounds != None:
                    f_area = 0
                    for ground in f_grounds:
                        x, y, dx, dy = list(ground.astype(np.int64))
                        f_area += dx*dy
                    area_sum += f_area
                    num_ground += 1
        return (area_sum/num_ground) / (320*240)