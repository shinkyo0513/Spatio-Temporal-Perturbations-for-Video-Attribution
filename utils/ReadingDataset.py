import os
from os.path import join, isdir, isfile

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch, torchvision

def loadTags(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        data = list(reader)
        
    tagName = [r[0] for r in data]
    return tagName, dict(zip(tagName, range(len(tagName))))

def getTagScore(scores, tags, tag2IDs):
    tagScore = []
    for r in tags:
        tagScore.append((r, scores[tag2IDs[r]]))
    return tagScore

def standardize_epic_video_name(name):
    video_name = name.split('/')[-1]
    prefix = video_name.split('-')[0].split('_')
    return '/'.join([prefix[0], '_'.join(prefix[:2]), video_name])

def standardize_ucf101_video_name(name):
    classification = name.split('_')[1]
    return '/'.join(['images', classification, name])

def get_frames(dataset_name, model_name, video_name, fids):
    if dataset_name == 'epic':
        vname = standardize_epic_video_name(video_name)
        root_path = f'{ds_root}/epic/seg_train'
        video_stf = int(sorted(os.listdir(os.path.join(root_path, vname)))[0][-14:-4])
        frames = [Image.open(os.path.join(root_path, vname, 
                        f'frame_{fid + video_stf:010d}.jpg')) for fid in fids]
    elif dataset_name == 'ucf101':
        root_path = f'{ds_root}/UCF101_24'
        frames = [Image.open(os.path.join(root_path, standardize_ucf101_video_name(video_name),
                        format(fid + 1, '05d') + '.jpg')) for fid in fids]
    elif dataset_name == 'cat_ucf':
        root_path = f'{ds_root}/Cat_UCF101'
        frames = [Image.open(os.path.join(root_path, standardize_ucf101_video_name(video_name),
                        format(fid + 1, '05d') + '.jpg')) for fid in fids]
    else:
        print('ERROR!')
        return
    
    if model_name == 'r2p1d' or model_name == 'r50l':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((112, 112)),
        ])
    elif model_name == 'v16l':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((240, 320)),
            torchvision.transforms.CenterCrop((224, 224))
        ])
    else:
        print('ERROR!')
        return
    
    return np.stack([np.array(transform(frame)) for frame in frames])

class FramesReader ():
    def __init__ (self, dataset_name, model_name):
        self.dataset_name = dataset_name
        self.model_name = model_name

        if dataset_name == 'epic':
            self.root_path = f'{ds_root}/epic/seg_train'
        elif dataset_name == 'ucf101':
            self.root_path = f'{ds_root}/UCF101_24'
        elif dataset_name == 'cat_ucf':
            self.root_path = f'{ds_root}/Cat_UCF101'
        else:
            print('ERROR!')
            return
        
        if model_name == 'r2p1d' or model_name == 'r50l':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((112, 112)),
            ])
        elif model_name == 'v16l':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((240, 320)),
                torchvision.transforms.CenterCrop((224, 224))
            ])
        else:
            print('ERROR!')
            return

    def get_frames (self, video_name, fids):
        if self.dataset_name == 'epic':
            vname = standardize_epic_video_name(video_name)
            video_stf = int(sorted(os.listdir(os.path.join(self.root_path, vname)))[0][-14:-4])
            frames = [Image.open(os.path.join(self.root_path, vname, 
                            f'frame_{fid + video_stf:010d}.jpg')) for fid in fids]
        elif self.dataset_name == 'ucf101':
            frames = [Image.open(os.path.join(self.root_path, standardize_ucf101_video_name(video_name),
                            format(fid + 1, '05d') + '.jpg')) for fid in fids]
        elif self.dataset_name == 'cat_ucf':
            frames = [Image.open(os.path.join(self.root_path, standardize_ucf101_video_name(video_name),
                            format(fid + 1, '05d') + '.jpg')) for fid in fids]
        else:
            print('ERROR!')
            return
        
        return np.stack([np.array(self.transform(frame)) for frame in self.frames])


def load_model_and_dataset (dataset_name, model_name, phase='val'):
    assert dataset_name in ['ucf101', 'epic', 'cat_ucf']
    assert model_name in ['r2p1d', 'r50l']
    if isinstance(phase, list):
        assert phase in [['val'], ['train'], ['val', 'train'], ['train', 'val']]
    else:
        assert phase in ['val', 'train']

    if dataset_name == "ucf101":
        num_classes = 24
        ds_path = f'{ds_root}/UCF101_24'
        if model_name == "r2p1d":
            from datasets.ucf101_24_dataset_new import UCF101_24_Dataset as dataset
            from model_def.r2plus1d import r2plus1d as model
            # model_wgts_dir = f"{proj_root}/model_param/ucf101_24_r2plus1d_18_16_Full_LongRange.pt"
            model_wgts_dir = f"{proj_root}/model_param/ucf101_24_r2p1d_16_Full_LongRange.pt"
        elif model_name == "r50l":
            from datasets.ucf101_24_dataset_new import UCF101_24_Dataset as dataset
            from model_def.r50lstm import r50lstm as model
            model_wgts_dir = f"{proj_root}/model_param/ucf101_24_r50l_16_Full_LongRange.pt"
        elif model_name == "v16l":
            from datasets.ucf101_24_dataset_vgg16lstm import UCF101_24_Dataset as dataset
            from model_def.vgg16lstm import vgg16lstm as model
            model_wgts_dir = f"{proj_root}/model_param/ucf101_24_vgg16lstm_16_Full_LongRange.pt"
    elif dataset_name == "epic":
        num_classes = 20
        ds_path = os.path.join(ds_root, path_dict.epic_rltv_dir)
        if model_name == "r2p1d":
            from datasets.epic_kitchens_dataset_new import EPIC_Kitchens_Dataset as dataset
            from model_def.r2plus1d import r2plus1d as model
            model_wgts_dir = f"{proj_root}/model_param/epic_r2p1d_16_Full_LongRange.pt"
        elif model_name == "r50l":
            from datasets.epic_kitchens_dataset_new import EPIC_Kitchens_Dataset as dataset
            from model_def.r50lstm import r50lstm as model
            model_wgts_dir = f"{proj_root}/model_param/epic_r50l_16_Full_LongRange.pt"
        elif model_name == "v16l":
            from datasets.epic_kitchens_dataset_vgg16lstm import EPIC_Kitchens_Dataset as dataset
            from model_def.vgg16lstm import vgg16lstm as model
            model_wgts_dir = f"{proj_root}/model_param/epic_vgg16lstm_16_Full_LongRange.pt"
    elif dataset_name == "cat_ucf":
        num_classes = 24
        ds_path = f'{ds_root}/Cat_UCF101'
        if model_name == "r2p1d":
            from datasets.cat_ucf_testset_new import Cat_UCF_Testset as dataset
            from model_def.r2plus1d import r2plus1d as model
            model_wgts_dir = f"{proj_root}/model_param/ucf101_24_r2p1d_16_Full_LongRange.pt"
        elif model_name == "r50l":
            from datasets.cat_ucf_testset_new import Cat_UCF_Testset as dataset
            from model_def.r50lstm import r50lstm as model
            model_wgts_dir = f"{proj_root}/model_param/ucf101_24_r50l_16_Full_LongRange.pt"

    if model_name == "r2p1d" or model_name == "r50l":
        model_ft = model(num_classes=num_classes, with_softmax=True)
    elif model_name == "v16l":
        model_ft = model(num_classes=num_classes)
    model_ft.load_weights(model_wgts_dir)

    sample_mode = 'long_range_last'
    num_frame = 16
    if isinstance(phase, list):
        video_datasets = {x: dataset(ds_path, num_frame, sample_mode, 1, 6, \
                                x=='train', testlist_idx=1) for x in phase}
        # print(rank, {x: 'Num of clips:{}'.format(len(video_datasets[x])) for x in ['train', 'val']})
        return model_ft, video_datasets
    else:
        video_dataset = dataset(ds_path, num_frame, sample_mode, 1, 6, phase=='train', testlist_idx=1)
        return model_ft, video_dataset