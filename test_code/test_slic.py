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
imgs1 = get_frames('ucf101', 'r2p1d', 'v_HorseRiding_g01_c01', [0,10,20,30])
imgs2 = get_frames('ucf101', 'r2p1d', 'v_SalsaSpin_g01_c02', [0,10,20,30])
imgs = np.concatenate([imgs1, imgs2], axis=0)
# img = imgs[0]
segments_slic = slic(imgs, n_segments=200, compactness=25, sigma=1)
print(imgs.shape, segments_slic.shape)
print(f"SLIC number of segments: {len(np.unique(segments_slic))}")

fig, axes = plt.subplots(1, 8, figsize=(28, 4), sharex=True, sharey=True)
for aidx, ax in enumerate(axes):
    ax.imshow(mark_boundaries(imgs[aidx], segments_slic[aidx]))
    # ax.set_title('SLIC')

plt.tight_layout()
fig.savefig('test.png')