import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import io
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import grey_dilation, grey_erosion
from scipy.segmentation import slic

from utils.CalAcc import process_activations
from utils.ImageShow import *

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

def plot_causal_metric_curve (scores, all_steps=None, show_txt=None, save_dir=None, return_img=False, dpi=75):
    assert isinstance(scores, np.ndarray) and len(scores.shape)==1
    num_score = scores.shape[0]
    print(num_score)

    if all_steps == None:
        all_steps = num_score
    # else:
    #     all_steps += 1
    #     assert all_steps >= num_score

    x_coords = np.arange(num_score) / all_steps
    fig = plt.figure(figsize=(8,8))
    plt.subplot(111)
    plt.plot(x_coords, scores)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.fill_between(x_coords, scores, alpha=0.4)

    if show_txt is not None:
        plt.title(show_txt)

    if save_dir is not None:
        plt.savefig(save_dir)

    if return_img:
        img = get_img_from_fig(fig, dpi=dpi)
        plt.close()
        return img
    plt.close()

def ani_update(i, frames1, frames2, axs):
    if i != 0:
        plt.cla() 
    for ax_idx, ax in enumerate(axs):
        if ax_idx == 0:
            ax.imshow(frames1[i])
            # ax.set_title("original frames", fontsize=14)
        else:
            ax.imshow(frames2[i])
            # ax.set_title(meth_name_dict[meth_names[ax_idx-1]], fontsize=14)
        ax.axis('off')

    # plt.subplots_adjust(left=0.01,
    #                 bottom=0.01, 
    #                 right=0.99, 
    #                 top=0.99, 
    #                 # wspace=0.2, 
    #                 hspace=0.01)
    # plt.tight_layout()

class CausalMetric():
    def __init__ (self, model, device):
        self.model = model
        self.device = device

    def coarsly_evaluate(self, mode, clip_batch, exp_batch, class_ids, 
                        remove_method="fade", new_size=None, n_step=1024, keep_topk=1.0,
                        visualize=False, video_names=None, vis_dir=None, denoise=False):
        assert mode == "del" or mode == "ins"

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
            exp_tensor = exp_batch.to(self.device)
        else:
            raise Exception(f"Given type of explanation is wrong, should be torch.tensor or np.ndarray")

        bs, nc, nt, nrow, ncol = clip_tensor.shape
        assert list(exp_tensor.shape) == [bs, 1, nt, nrow, ncol]
        assert nrow == ncol

        if denoise:
            exp_np = exp_tensor.cpu().numpy()
            denoised_exp_np = grey_dilation(grey_erosion(exp_np, (1,1,1,7,7)), (1,1,1,7,7))
            exp_tensor = torch.from_numpy(denoised_exp_np).to(self.device)

        if remove_method == "fade":
            base = torch.zeros_like(clip_tensor).to(self.device)    # N x C x T x H x W
        elif remove_method == "blur":
            base = blur(clip_tensor, klen=11, ksig=5, device=self.device)    # N x C x T x H x W

        if new_size == None or new_size == nrow:
            small_exp_tensor = exp_tensor.transpose(1,2).contiguous()
            new_size = nrow
        else:
            assert nrow % new_size == 0
            ks = nrow // new_size
            k = torch.ones((1, 1, ks, ks)).to(self.device) / (ks*ks)
            exp_tensor = exp_tensor.reshape((-1, nrow, ncol)).unsqueeze(1)  # N*T x 1 x H x W
            small_exp_tensor = F.conv2d(exp_tensor, k, stride=ks, padding=0)    # N*T x 1 x H' x W'
            small_exp_tensor = small_exp_tensor.reshape(bs, nt, 1, new_size, new_size)    # N x T x 1 x H' x W'
        
        volume = nt * new_size**2
        step = int(volume // n_step)

        sal_order = small_exp_tensor.reshape(bs, -1).argsort(dim=-1, descending=True) # N x T*H'*W'
        idx = torch.arange(bs, dtype=torch.long).view(-1,1).to(self.device)

        if keep_topk < 1.0 and keep_topk > 0.0:
            assert isinstance(keep_topk, float) and keep_topk >= 0.0
            num_topk = int(volume * keep_topk)
            num_zeros = volume - num_topk
            rand_order = torch.stack([torch.randperm(num_zeros) for bidx in range(bs)], dim=0).to(self.device)
            sal_order = torch.cat([
                            sal_order[:, :num_topk], 
                            sal_order[:, num_topk:][idx, rand_order]
                        ], dim=1)
        elif keep_topk == 0.0:
            sal_order = torch.stack([torch.randperm(volume) for bidx in range(bs)], dim=0).to(self.device)

        if mode == "del":
            pmask = torch.ones(bs, nt, new_size, new_size).to(self.device)
        elif mode == "ins":
            pmask = torch.zeros(bs, nt, new_size, new_size).to(self.device)

        prob_lst = []
        max_prob = 0
        max_mask = None
        if visualize:
            video_vis_list = [list(),] * bs
            prob_vis_list = [list(),] * bs
        # for t in range(0, volume, step):
            # pos = sal_order[:, t:t+step]  # N*T x 1 x 1
        for t in range(n_step+1):
            if t == 0:
                pmask = pmask
            else:
                st = (t-1) * step
                ed = min(t * step, volume)
                pos = sal_order[:, st:ed]  # N*T x 1 x 1
                if mode == 'del':
                    pmask.reshape(bs, -1)[idx, pos] = 0
                elif mode == 'ins':
                    pmask.reshape(bs, -1)[idx, pos] = 1

            mask = F.interpolate(pmask, size=(nrow, ncol), mode='nearest').unsqueeze(1)  # N x 1 x T x H x W
            perturb_clip = clip_tensor * mask + base * (1 - mask)
            y = self.model(perturb_clip)
            prob, pred_label, pred_label_prob = process_activations(y, class_ids, softmaxed=True)
            prob_lst.append(prob.detach().cpu())

            if visualize:
                perturb_clip_cpu = perturb_clip.detach().cpu()
                mask_cpu = mask.detach().cpu()
                prob_lst_cpu = torch.stack(prob_lst, dim=-1)
                for bidx in range(bs):
                    video_vis_np = plot_voxel(perturb_clip_cpu[bidx], mask_cpu[bidx], return_img=True, dpi=150, 
                                                title=f"{video_names[bidx]} #{t}/{n_step} P={prob_lst_cpu[bidx][-1]:.3f}")
                    video_vis_list[bidx].append(video_vis_np)
                    prob_vis_np = plot_causal_metric_curve(prob_lst_cpu[bidx].numpy(), all_steps=n_step, return_img=True, dpi=150)
                    prob_vis_list[bidx].append(prob_vis_np)
                    print(video_vis_np.shape, prob_vis_np.shape, f"{prob_lst_cpu[bidx][-1]:.3f}")

        if visualize:
            for bidx in range(bs):
                fig, axs = plt.subplots(1, 2, figsize=(22, 8), gridspec_kw={'width_ratios': [2, 1]}, 
                                                constrained_layout=True)
                ani = animation.FuncAnimation(fig, ani_update, fargs=(video_vis_list[bidx], prob_vis_list[bidx], axs), 
                                                interval=1000, frames=len(video_vis_list[bidx]))
                if video_names != None and vis_dir != None:
                    video_name = video_names[bidx].split('/')[-1]
                    ani.save(f'{vis_dir}/{video_name}.gif', writer="imagemagick")
                
        prob_lst = torch.stack(prob_lst, dim=-1)    # N x T*H'*W'
        return prob_lst



class NewCausalMetric():
    def __init__ (self, model, device):
        self.model = model
        self.device = device

    def coarsly_evaluate(self, mode, clip_batch, exp_batch, class_ids, 
                        remove_method="fade", new_size=None, n_step=1024, keep_topk=1.0,
                        visualize=False, video_names=None, vis_dir=None, denoise=False, 
                        superpixel=False):
        assert mode == "del" or mode == "ins"

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
            exp_tensor = exp_batch.to(self.device)
        else:
            raise Exception(f"Given type of explanation is wrong, should be torch.tensor or np.ndarray")

        bs, nc, nt, nrow, ncol = clip_tensor.shape
        assert list(exp_tensor.shape) == [bs, 1, nt, nrow, ncol]
        assert nrow == ncol

        if denoise:
            exp_np = exp_tensor.cpu().numpy()
            denoised_exp_np = grey_dilation(grey_erosion(exp_np, (1,1,1,7,7)), (1,1,1,7,7))
            exp_tensor = torch.from_numpy(denoised_exp_np).to(self.device)

        if remove_method == "fade":
            base = torch.zeros_like(clip_tensor).to(self.device)    # N x C x T x H x W
        elif remove_method == "blur":
            base = blur(clip_tensor, klen=11, ksig=5, device=self.device)    # N x C x T x H x W

        if new_size == None or new_size == nrow:
            small_exp_tensor = exp_tensor.transpose(1,2).contiguous()
            new_size = nrow
        else:
            assert nrow % new_size == 0
            ks = nrow // new_size
            k = torch.ones((1, 1, ks, ks)).to(self.device) / (ks*ks)
            exp_tensor = exp_tensor.reshape((-1, nrow, ncol)).unsqueeze(1)  # N*T x 1 x H x W
            small_exp_tensor = F.conv2d(exp_tensor, k, stride=ks, padding=0)    # N*T x 1 x H' x W'
            small_exp_tensor = small_exp_tensor.reshape(bs, nt, 1, new_size, new_size)    # N x T x 1 x H' x W'
        
        volume = nt * new_size**2
        step = int(volume // n_step)

        sal_order = small_exp_tensor.reshape(bs, -1).argsort(dim=-1, descending=True) # N x T*H'*W'
        idx = torch.arange(bs, dtype=torch.long).view(-1,1).to(self.device)

        if keep_topk < 1.0 and keep_topk > 0.0:
            assert isinstance(keep_topk, float) and keep_topk >= 0.0
            num_topk = int(volume * keep_topk)
            num_zeros = volume - num_topk
            rand_order = torch.stack([torch.randperm(num_zeros) for bidx in range(bs)], dim=0).to(self.device)
            sal_order = torch.cat([
                            sal_order[:, :num_topk], 
                            sal_order[:, num_topk:][idx, rand_order]
                        ], dim=1)
        elif keep_topk == 0.0:
            sal_order = torch.stack([torch.randperm(volume) for bidx in range(bs)], dim=0).to(self.device)

        if mode == "del":
            pmask = torch.ones(bs, nt, new_size, new_size).to(self.device)
        elif mode == "ins":
            pmask = torch.zeros(bs, nt, new_size, new_size).to(self.device)

        prob_lst = []
        max_prob = 0
        max_mask = None
        if visualize:
            video_vis_list = [list(),] * bs
            prob_vis_list = [list(),] * bs
        # for t in range(0, volume, step):
            # pos = sal_order[:, t:t+step]  # N*T x 1 x 1
        for t in range(n_step+1):
            if t == 0:
                pmask = pmask
            else:
                st = (t-1) * step
                ed = min(t * step, volume)
                pos = sal_order[:, st:ed]  # N*T x 1 x 1
                if mode == 'del':
                    pmask.reshape(bs, -1)[idx, pos] = 0
                elif mode == 'ins':
                    pmask.reshape(bs, -1)[idx, pos] = 1

            mask = F.interpolate(pmask, size=(nrow, ncol), mode='nearest').unsqueeze(1)  # N x 1 x T x H x W
            perturb_clip = clip_tensor * mask + base * (1 - mask)
            y = self.model(perturb_clip)
            prob, pred_label, pred_label_prob = process_activations(y, class_ids, softmaxed=True)
            prob_lst.append(prob.detach().cpu())

            if visualize:
                perturb_clip_cpu = perturb_clip.detach().cpu()
                mask_cpu = mask.detach().cpu()
                prob_lst_cpu = torch.stack(prob_lst, dim=-1)
                for bidx in range(bs):
                    video_vis_np = plot_voxel(perturb_clip_cpu[bidx], mask_cpu[bidx], return_img=True, dpi=150, 
                                                title=f"{video_names[bidx]} #{t}/{n_step} P={prob_lst_cpu[bidx][-1]:.3f}")
                    video_vis_list[bidx].append(video_vis_np)
                    prob_vis_np = plot_causal_metric_curve(prob_lst_cpu[bidx].numpy(), all_steps=n_step, return_img=True, dpi=150)
                    prob_vis_list[bidx].append(prob_vis_np)
                    print(video_vis_np.shape, prob_vis_np.shape, f"{prob_lst_cpu[bidx][-1]:.3f}")

        if visualize:
            for bidx in range(bs):
                fig, axs = plt.subplots(1, 2, figsize=(22, 8), gridspec_kw={'width_ratios': [2, 1]}, 
                                                constrained_layout=True)
                ani = animation.FuncAnimation(fig, ani_update, fargs=(video_vis_list[bidx], prob_vis_list[bidx], axs), 
                                                interval=1000, frames=len(video_vis_list[bidx]))
                if video_names != None and vis_dir != None:
                    video_name = video_names[bidx].split('/')[-1]
                    ani.save(f'{vis_dir}/{video_name}.gif', writer="imagemagick")
                
        prob_lst = torch.stack(prob_lst, dim=-1)    # N x T*H'*W'
        return prob_lst
