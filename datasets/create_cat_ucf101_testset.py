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
from tqdm import tqdm

import accimage
torchvision.set_image_backend('accimage')
from accimage import Image

import sys
sys.path.append(".")
sys.path.append("..")
from utils.GaussianSmoothing import GaussianSmoothing
from utils import LongRangeSample 
from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

if __name__ == "__main__":
    ucf101_dir = f"{ds_root}/UCF101_24"
    annot_file = join(ucf101_dir, 'annotations/pyannot.pkl')
    split_file = join(ucf101_dir, 'splits/testlist01.txt')
    rgb_file = join(ucf101_dir, 'images')

    video_list = open(split_file, 'r').read().splitlines()

    video_numf_dict = {}
    for video_name in video_list:
        video_path = join(rgb_file, video_name)
        frame_names = [frame_name for frame_name in listdir(video_path) if "jpg" in frame_name]
        num_frame = len(frame_names)
        video_numf_dict[video_name] = num_frame
    sorted_video_list = [k for k, v in sorted(video_numf_dict.items(), key=lambda item: item[1])]

    with open(annot_file, 'rb') as f:
        # video_annot_dict[video_name]: (['annotations', 'numf', 'label'])
        all_video_annot_dict = pickle.load(f)
    video_annot_dict = {video_name: all_video_annot_dict[video_name] for video_name in video_list}
    # video_annot_dict = {k: v for k, v in sorted(video_annot_dict.items(), key=lambda item: int(item[1]['numf']))}
    # sorted_video_list = list(video_annot_dict.keys())
    # for idx, video in enumerate(sorted_video_list[:20]):
    #     print(f"{idx}: {video}")

    random_video_pair = {}
    random.seed(0)
    for video_name in video_list:
        # video_numf = int(video_annot_dict[video_name]['numf'])
        video_numf = video_numf_dict[video_name]
        video_label = video_annot_dict[video_name]['label']
        video_sorted_idx = sorted_video_list.index(video_name)
        print(f'{video_name}: {video_numf} frames, {video_sorted_idx} index, {video_label} label...')

        if video_label == 1 and video_sorted_idx < 10: # BasketballDunk
                random_video_name = "Basketball/v_Basketball_g02_c01"
        else:
            choice_list = [v for v in sorted_video_list[:video_sorted_idx] if video_annot_dict[v]['label']!=video_label]
            random_video_name = random.choice(choice_list)

        random_video_pair[video_name] = random_video_name
        print(f'\tFound {random_video_name}!')

    frames_per_clip = 16
    cat_ucf101_dir = join(ds_root, 'Cat_UCF101')
    cat_rgb_file = join(ds_root, 'Cat_UCF101', 'images')
    cat_annot_file = join(ds_root, 'Cat_UCF101', 'UCF101_Cat_Pair.txt')
    os.makedirs(cat_rgb_file, exist_ok=True)

    with open(cat_annot_file, 'w') as f:
        for video_name, pair_video_name in list(random_video_pair.items()):
            os.makedirs(join(cat_rgb_file, video_name), exist_ok=True)

            side = random.choice(['left', 'right'])

            # numf = int(video_annot_dict[video_name]['numf'])
            # pair_numf = int(video_annot_dict[pair_video_name]['numf'])
            numf = video_numf_dict[video_name]
            pair_numf = video_numf_dict[pair_video_name]

            cl = int( frames_per_clip * numf / (numf + pair_numf) )
            fidxs = LongRangeSample.long_range_last(numf, cl)
            # print(f'{video_name}: {numf}, {fidxs}')
            
            pair_cl = frames_per_clip - cl
            pair_fidxs = LongRangeSample.long_range_last(pair_numf, pair_cl)
            # print(f'{pair_video_name}: {pair_numf}, {pair_fidxs}')
            
            video_name_order = [video_name, pair_video_name] if side == 'left' else [pair_video_name, video_name]
            cat_fidxs = fidxs + pair_fidxs if side == 'left' else pair_fidxs + fidxs
            exchange_cidx = cl if side == 'left' else pair_cl
            for cidx in range(frames_per_clip):
                if cidx < exchange_cidx:
                    src_video_name = video_name_order[0]
                else:
                    src_video_name = video_name_order[1]
                dst = join(cat_rgb_file, video_name, format(cidx+1, '05d')+'.jpg')
                src = join(rgb_file, src_video_name, format(cat_fidxs[cidx]+1, '05d')+'.jpg')
                os.symlink(src, dst)
            
            f.write(f'{video_name} {pair_video_name} {side} {cl}\n')
    f.close()

    print('Found all pairs. Start to save...')


class Cat_UCF_Test_Dataset (Dataset):
    def __init__ (self, root_dir, frames_per_clip=16, sample_mode='long_range_last', 
                    num_clips=1, frame_rate=6, train=False, testlist_idx=1):
        self.root_dir = root_dir
        self.annot_file = join(root_dir, 'annotations/pyannot.pkl')
        # Change split_file for smaller test set
        if testlist_idx == 1:
            self.split_file = join(root_dir, 'splits/testlist01.txt')
        elif testlist_idx == 2:
            self.split_file = join(root_dir, 'splits/testlist02.txt')
        self.rgb_file = join(root_dir, 'images')
        self.train = train

        self.frames_per_clip = frames_per_clip
        self.frame_rate = frame_rate
        self.sample_mode = sample_mode
        self.num_clips = num_clips

        with open(self.annot_file, 'rb') as f:
            # video_annot_dict[video_name]: (['annotations', 'numf', 'label'])
            self.video_annot_dict = pickle.load(f)

        self.test_list = open(self.split_file, 'r').read().splitlines()
        self.test_list = sorted(self.test_list)

        self.video_numf_dict = {}
        self.random_video_pair = {}
        random.seed(0)
        for video_name in self.test_list:
            video_path = join(self.rgb_file, video_name)
            frame_names = [frame_name for frame_name in listdir(video_path) if "jpg" in frame_name]
            num_frame = len(frame_names)
            self.video_numf_dict[video_name] = num_frame

            while True:
                random_video_name = random.choice(self.test_list)
                if self.video_annot_dict[video_name]['label'] != self.video_annot_dict[random_video_name]['label']:
                    # if int(self.video_annot_dict[video_name]['numf']) >= int(self.video_annot_dict[random_video_name]['numf']):
                    # print(f"{video_name}: {self.video_annot_dict[video_name]['numf']}; {random_video_name}: {self.video_annot_dict[random_video_name]['numf']}")
                    # print(type(int(self.video_annot_dict[video_name]['numf'])))
                    # print(f'{video_name} - {random_video_name}')
                    self.random_video_pair[video_name] = random_video_name
                    break

        self.transform = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
        ])

    def __len__ (self):
        return len(self.test_list)

    def __getitem__ (self, idx):
        video_name = self.test_list[idx]
        label = self.video_annot_dict[video_name]['label']
        numf = self.video_numf_dict[video_name]

        pair_video_name = self.random_video_pair[video_name]
        pair_numf = self.video_numf_dict[pair_video_name]

        cl = int( self.frames_per_clip * numf / (numf + pair_numf) )
        fidxs = LongRangeSample.long_range_last(numf, cl)
        clip_frames = [Image(join(self.rgb_file, video_name, format(fidx+1, '05d')+'.jpg')) for fidx in fidxs]

        pair_cl = self.frames_per_clip - cl
        pair_fidxs = LongRangeSample.long_range_last(pair_numf, pair_cl)
        pair_clip_frames = [Image(join(self.rgb_file, pair_video_name, format(fidx+1, '05d')+'.jpg')) for fidx in pair_fidxs]

        side = random.choice(['left', 'right'])
        if side == 'left':
            cat_clip_frames = clip_frames + pair_clip_frames
            ground_fidxs = [1,] * len(clip_frames) + [0,] * len(pair_clip_frames)
        elif side == 'right':
            cat_clip_frames = pair_clip_frames + clip_frames
            ground_fidxs = [0,] * len(pair_clip_frames) + [1,] * len(clip_frames)

        cat_clip_tensor = torch.stack([self.transform(cat_clip_frame) for cat_clip_frame in 
                                    cat_clip_frames], dim=1)    # 3x16x112x112
        ground_fidxs = torch.tensor(ground_fidxs).long()   # clip_len

        # print(len(clip_frames), len(pair_clip_frames), cat_clip_tensor.shape, ground_fidxs.shape)

        return cat_clip_tensor, label, video_name, ground_fidxs
