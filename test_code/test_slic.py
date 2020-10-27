import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import torch, torchvision

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ReadingDataset import get_frames

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

# img = img_as_float(astronaut()[::2, ::2])
imgs1 = get_frames('ucf101', 'r2p1d', 'v_HorseRiding_g01_c01', [0, 200])
imgs2 = get_frames('ucf101', 'r2p1d', 'v_SalsaSpin_g01_c02', [0, 100])
imgs3 = get_frames('epic', 'r2p1d', 'P06_09_11691-open-drawer', [0, 100])
imgs = np.concatenate([imgs1], axis=0)
# imgs = imgs1
# img = imgs[0]
segments_slic = slic(imgs, n_segments=256, sigma=1, slic_zero=True)#, compactness=20)
print(imgs.shape, segments_slic.shape)
print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
# print(segments_slic)

num_f = imgs.shape[0]
if num_f > 1:
    fig, axes = plt.subplots(1, num_f, figsize=(4*num_f, 4), sharex=True, sharey=True)
    for aidx, ax in enumerate(axes):
        ax.imshow(mark_boundaries(imgs[aidx], segments_slic[aidx]))
else:
    fig, ax = plt.subplots(1, num_f, figsize=(4*imgs.shape[0], 4), sharex=True, sharey=True)
    ax.imshow(mark_boundaries(imgs[0], segments_slic[0]))

plt.tight_layout()
fig.savefig('test256.png')