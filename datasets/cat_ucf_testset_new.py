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
import copy

import accimage
torchvision.set_image_backend('accimage')
from accimage import Image

import sys
sys.path.append("..")
from utils.GaussianSmoothing import GaussianSmoothing
from utils import LongRangeSample 


class Cat_UCF_Testset (Dataset):
    def __init__ (self, root_dir, frames_per_clip=16, sample_mode='long_range_last', 
                    num_clips=1, frame_rate=6, train=False, testlist_idx=1):
        self.root_dir = root_dir
        self.pair_file = join(root_dir, 'UCF101_Cat_Pair.txt')
        self.annot_file = join(root_dir, 'pyannot.pkl')
        # Change split_file for smaller test set
        # if testlist_idx == 1:
        #     self.split_file = join(root_dir, 'splits/testlist01.txt')
        # elif testlist_idx == 2:
        #     self.split_file = join(root_dir, 'splits/testlist02.txt')
        self.rgb_file = join(root_dir, 'images')

        self.frames_per_clip = frames_per_clip
        self.frame_rate = frame_rate
        self.sample_mode = sample_mode
        self.num_clips = num_clips

        with open(self.pair_file, 'r') as f:
            self.video_pair_dict = {}
            for line in f.readlines():
                video, paired_video, gt_side, gt_numf = line.strip().split(' ')
                self.video_pair_dict[video] = {'paired_video': paired_video, 'gt_side': gt_side, 'gt_numf': gt_numf}
        f.close()

        with open(self.annot_file, 'rb') as f:
            # video_annot_dict[video_name]: (['annotations', 'numf', 'label'])
            self.video_annot_dict = pickle.load(f)
        f.close()

        self.video_list = list(self.video_pair_dict.keys())

        self.transform = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
        ])

    def __len__ (self):
        return len(self.video_list)

    def __getitem__ (self, idx):
        video_name = self.video_list[idx]
        label = self.video_annot_dict[video_name]['label']
        gt_side = self.video_pair_dict[video_name]['gt_side']
        gt_numf = self.video_pair_dict[video_name]['gt_numf']
        
        fidxs = list(range(16))
        clip_frames = [Image(join(self.rgb_file, video_name, format(fidx+1, '05d')+'.jpg')) for fidx in fidxs]
        clip_tensor = torch.stack([self.transform(clip_frame) for clip_frame in clip_frames], dim=1)    # 3x16x112x112

        # gt_fidxs = list(range(gt_numf)) if gt_side == 'left' else list(range(gt_numf, 16))
        # ground_fidxs = torch.tensor(ground_fidxs).long()   # clip_len
        fidxs_tensor = torch.tensor(list(range(16))).long()

        return clip_tensor, label, video_name, fidxs_tensor
