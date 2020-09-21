import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from utils.CalAcc import process_activations

# HW = 112 * 112 # image area
# THW = 112*112*16
# n_classes = 80
# blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)
def blur(x, klen=11, ksig=5, device=torch.device("cpu")):
    kern = gkern(klen, ksig).to(device)
    numf = x.shape[2]   # 16 or 8
    blur_x = []
    for fidx in range(numf):
        blur_x.append(nn.functional.conv2d(x[:,:,fidx,:,:], kern, padding=klen//2))
    blur_x = torch.stack(blur_x, dim=2)
    return blur_x

def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

def plot_causal_metric_curve (scores, show_txt=None, save_dir=None):
    assert isinstance(scores, np.ndarray) and len(scores.shape)==1
    num_score = scores.shape[0]

    x_coords = np.arange(num_score) / num_score
    plt.subplot(111)
    plt.plot(x_coords, scores)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.fill_between(x_coords, scores, alpha=0.4)

    if show_txt is not None:
        plt.title(show_txt)

    if save_dir is not None:
        plt.savefig(save_dir)

    plt.close()

def calulate_pearson (input, target):
    x = input
    y = target

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    p = torch.sum(vx*vy, dim=1) / (torch.sqrt(torch.sum(vx**2, dim=1)) * torch.sqrt(torch.sum(vy**2, dim=1)))
    return p

class Faithfulness():
    def __init__ (self, model, device):
        self.model = model
        self.device = device

    def calculate(self, clip_batch, exp_batch, class_ids, remove_method="fade"):
        # clip_tensor: N x C x T x H x W
        if isinstance(clip_batch, np.ndarray):
            clip_tensor = torch.from_numpy(clip_tensor).to(self.device)
        elif isinstance(clip_batch, torch.Tensor):
            clip_tensor = clip_batch.to(self.device)
        else:
            raise Exception(f"Given type of clip is wrong, should be torch.tensor or np.ndarray")

        # exp_tensor: N x 1 x T x H x W
        if isinstance(exp_batch, np.ndarray):
            exp_tensor = torch.from_numpy(exp_batch).to(self.device)
        elif isinstance(exp_batch, torch.Tensor):
            exp_tensor = exp_batch
        else:
            raise Exception(f"Given type of explanation is wrong, should be torch.tensor or np.ndarray")

        bs, nc, nt, nrow, ncol = clip_tensor.shape
        assert list(exp_tensor.shape) == [bs, 1, nt, nrow, ncol]
        assert nrow == ncol

        y = self.model(clip_tensor)
        prob, pred_label, pred_label_prob = process_activations(y, class_ids, softmaxed=True)
        prob0 = prob.detach().cpu()

        if remove_method == "fade":
            base = torch.zeros_like(clip_tensor).to(self.device)    # N x C x T x H x W
        elif remove_method == "blur":
            base = blur(clip_tensor, klen=11, ksig=5, device=self.device)    # N x C x T x H x W

        ks = 112
        small_size = nrow // ks
        k = torch.ones((1, 1, ks, ks)) / (ks*ks)
        exp_tensor = exp_tensor.reshape((-1, nrow, ncol)).unsqueeze(1)  # N*T x 1 x H x W
        small_exp_tensor = F.conv2d(exp_tensor, k, stride=ks, padding=0)    # N*T x 1 x H' x W'
        small_exp_tensor = small_exp_tensor.reshape(bs, nt, 1, small_size, small_size)    # N x T x 1 x H' x W'

        sorted_small_exp, sal_order = small_exp_tensor.reshape(bs, -1).sort(dim=-1, descending=True) # N x T*H'*W'
        idx = torch.arange(bs, dtype=torch.long).view(-1,1).to(self.device)
        
        pmask = torch.ones(bs, nt, small_size, small_size).to(self.device)

        delta_probs = []
        volume = nt * small_size**2
        for t in range(0, volume):
            pos = sal_order[:, t:t+1]
            pmask.reshape(bs, -1)[idx, pos] = 0
            mask = F.interpolate(pmask, size=(nrow, ncol), mode='nearest').unsqueeze(1)  # N x 1 x T x H x W
            perturb_clip = clip_tensor * mask + base * (1 - mask)
            pmask.reshape(bs, -1)[idx, pos] = 1

            y = self.model(perturb_clip)
            prob, pred_label, pred_label_prob = process_activations(y, class_ids, softmaxed=True)
            delta_probs.append( prob0 - prob.detach().cpu() )
        delta_probs = torch.stack(delta_probs, dim=-1)    # N x T*H'*W'
        # for bidx in range(bs):
        #     print(delta_probs[bidx])

        pearson_coeffs = calulate_pearson(delta_probs, sorted_small_exp.detach().cpu())
        return pearson_coeffs
