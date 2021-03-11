#%%
import os
import glob
import tqdm
import shutil
import ast
import math
from PIL import Image, ImageFile

import pandas as pd
import json
import cv2
import random
random.seed(1)

import sys
sys.path.append(".")
sys.path.append("..")

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

frames_dir = os.path.join(ds_root, 'something_something_v2/20bn-something-something-v2-frames')
annot_dir = os.path.join(ds_root, 'something_something_v2/annotations')
annot_labels_path = os.path.join(annot_dir, 'something-something-v2-labels.json')
annot_train_path = os.path.join(annot_dir, 'something-something-v2-train.json')
annot_val_path = os.path.join(annot_dir, 'something-something-v2-validation.json')

# video_frame_stats = {}
# for video_id in sorted(glob.glob(os.path.join(frames_dir, '*'))):
#     video_dir = os.path.join(frames_dir, video_id)
#     video_frames = glob.glob(os.path.join(video_dir, '*.jpg'))
#     num_frame = len(video_frames)
#     video_frame_stats[video_id] = num_frame

with open(annot_labels_path) as f:
    labels_dict = json.load(f)
labels_list = list(labels_dict.keys())

# Train
with open(annot_train_path) as f:
    train_list = json.load(f)

train_label_stats = {}
for train_sample in train_list:
    # action_label = train_sample['template'].replace("[", "").replace("]", "")
    action_label = train_sample['template']
    train_label_stats[action_label] = train_label_stats.get(action_label, 0) + 1
train_label_stats = {k: v for k, v in sorted(train_label_stats.items(), key=lambda item: item[1], reverse=True)}

# Validation
with open(annot_val_path) as f:
    val_list = json.load(f)

val_label_stats = {}
for val_sample in val_list:
    # action_label = val_sample['template'].replace("[", "").replace("]", "")
    action_label = val_sample['template']
    val_label_stats[action_label] = val_label_stats.get(action_label, 0) + 1
val_label_stats = {k: v for k, v in sorted(val_label_stats.items(), key=lambda item: item[1], reverse=True)}

frame_thres = 32
top25_labels = list(val_label_stats.keys())[:25]
print(top25_labels)

top25_labels_index = {label.replace('[', '').replace(']', ''): index for index, label in enumerate(top25_labels)}
top25_labels_index_path = os.path.join(proj_root, 'my_sthsthv2_annot', 'top25_labels_index_1.json')
with open(top25_labels_index_path, 'w') as f:
    json.dump(top25_labels_index, f)

# Val Selection
val_500_samples = []
val_500_stats = {}
val_1000_samples = []
val_1000_stats = {}

random.shuffle(val_list)
for val_sample in val_list:
    action_label = val_sample['template']
    if action_label in top25_labels:
        video_id = val_sample['id']
        num_frames = len(glob.glob(os.path.join(frames_dir, video_id, '*.jpg')))
        if num_frames > frame_thres:
            if val_500_stats.get(action_label, 0) < 20:
                val_500_samples.append(val_sample)
                val_500_stats[action_label] = val_500_stats.get(action_label, 0) + 1
            if val_1000_stats.get(action_label, 0) < 40:
                val_1000_samples.append(val_sample)
                val_1000_stats[action_label] = val_1000_stats.get(action_label, 0) + 1
            
for action_label in val_500_stats.keys():
    print(action_label, val_500_stats[action_label], val_1000_stats[action_label])

val_500_path = os.path.join(proj_root, 'my_sthsthv2_annot', 'val_500_1.json')
with open(val_500_path, 'w') as f:
    json.dump(val_500_samples, f)

val_1000_path = os.path.join(proj_root, 'my_sthsthv2_annot', 'val_1000_1.json')
with open(val_1000_path, 'w') as f:
    json.dump(val_1000_samples, f)

# Train Selection
train_7000_samples = []
train_7000_stats = {}
random.shuffle(train_list)
for train_sample in train_list:
    action_label = train_sample['template']
    if action_label in top25_labels:
        video_id = train_sample['id']
        num_frames = len(glob.glob(os.path.join(frames_dir, video_id, '*.jpg')))
        if num_frames > frame_thres:
            if train_7000_stats.get(action_label, 0) < 280:
                train_7000_samples.append(train_sample)
                train_7000_stats[action_label] = train_7000_stats.get(action_label, 0) + 1

for action_label in train_7000_stats.keys():
    print(action_label, train_7000_stats[action_label])

train_7000_path = os.path.join(proj_root, 'my_sthsthv2_annot', 'train_7000_1.json')
with open(train_7000_path, 'w') as f:
    json.dump(train_7000_samples, f)