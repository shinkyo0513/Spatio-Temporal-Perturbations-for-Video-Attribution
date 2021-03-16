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
from skimage.filters import gaussian
# from scipy.segmentation import slic

from utils.CalAcc import process_activations
from utils.ImageShow import *
from utils.ReadingDataset import get_frames
from utils.TensorSmooth import imsmooth

def gaussian_blur(x, ksig=5, klen=11, device=torch.device('cpu')):
    bs, nc, nt, h, w = x.shape
    kern = gkern(ksig, klen, nc)
    kern = kern.to(device)
    blur_x = []
    for fidx in range(nt):
        blur_x.append(nn.functional.conv2d(x[:,:,fidx,:,:], kern, padding=klen//2))
    blur_x = torch.stack(blur_x, dim=2)
    # print(blur_x.shape)
    return blur_x

def gkern(nsig, klen=11, num_ch=3):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((num_ch, num_ch, klen, klen))
    for ch in range(num_ch):
        kern[ch, ch] = k
    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

def plot_causal_metric_curve (scores, all_steps=None, show_txt=None, save_dir=None, return_img=False, dpi=75):
    assert isinstance(scores, np.ndarray) and len(scores.shape)==1
    num_score = scores.shape[0]
    # print(num_score-1)

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

    plt.xlabel('Pixels Perturbed (%)')
    plt.ylabel('Probability (%)')

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

    def evaluate(self, mode, clip_batch, exp_batch, class_ids, 
                order="most_first", remove_method="fade", new_size=None, n_step=1024, 
                keep_topk=1.0, vis_process=False, vis_dir=None, video_names=None, 
                mask_smooth_sigma=0):
        assert mode == "del" or mode == "ins"
        assert order == "most_first" or order == "least_first"

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

        if mask_smooth_sigma != 0:
            exp_tensor = gaussian_blur(exp_tensor, ksig=mask_smooth_sigma, 
                            klen=2*mask_smooth_sigma+1, device=self.device)   # N x 1 x T x H x W

        if remove_method == "fade":
            base = torch.zeros_like(clip_tensor).to(self.device)    # N x C x T x H x W
        elif remove_method == "blur":
            # base = gaussian_blur(clip_tensor, klen=11, ksig=5, device=self.device)    # N x C x T x H x W
            reshaped_clip_tensor = torch.cat(torch.unbind(clip_tensor, dim=2), dim=0)   # T*N x C x H x W
            base = imsmooth(reshaped_clip_tensor, sigma=20)    # T*N x C x H x W
            base = torch.stack(base.split(bs, dim=0), dim=2)   # N x C x T x H x W

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

        if order == "most_first":
            sal_order = small_exp_tensor.reshape(bs, -1).argsort(dim=-1, descending=True) # N x T*H'*W'
        elif order == "least_first":
            sal_order = small_exp_tensor.reshape(bs, -1).argsort(dim=-1, descending=False) # N x T*H'*W'
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
        if vis_process:
            video_vis_list = [list(),] * bs
            prob_vis_list = [list(),] * bs
        step_pts = np.linspace(0, volume, n_step+1).astype(np.int).tolist()
        for t in range(n_step+1):
            if t == 0:
                pmask = pmask
            else:
                st = step_pts[t-1]
                ed = step_pts[t]
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

            if vis_process:
                perturb_clip_cpu = perturb_clip.detach().cpu()
                mask_cpu = mask.detach().cpu()
                prob_lst_cpu = torch.stack(prob_lst, dim=-1)
                for bidx in range(bs):
                    video_vis_np = plot_voxel(perturb_clip_cpu[bidx], mask_cpu[bidx], return_img=True, dpi=150, 
                                                title=f"{video_names[bidx]} #{t}/{n_step} P={prob_lst_cpu[bidx][-1]:.3f}")
                    video_vis_list[bidx].append(video_vis_np)
                    prob_vis_np = plot_causal_metric_curve(prob_lst_cpu[bidx].numpy(), all_steps=n_step, return_img=True, dpi=150)
                    prob_vis_list[bidx].append(prob_vis_np)

                    mask_area = mask_cpu[bidx].sum().item() / volume
                    print(video_vis_np.shape, prob_vis_np.shape, f"Acc={prob_lst_cpu[bidx][-1]:.3f}", f"S={mask_area:.3f}")

        if vis_process:
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


    def evaluate_by_superpixel(self, mode, clip_tensor, exp_tensor, class_id, superpixel_mask,
                                order = "most_first", remove_method="fade", n_step=1024, mask_smooth_sigma=0, 
                                parallel_size=7, vis_process=False, vis_dir=None, video_name=None):
        assert mode == "del" or mode == "ins"
        assert order == "most_first" or order == "least_first"
        assert len(clip_tensor.shape) == 4, f'Do not support batched input! Input shape: {clip_tensor.shape}'

        # clip_tensor: C x T x H x W
        if isinstance(clip_tensor, np.ndarray):
            clip_tensor = torch.from_numpy(clip_tensor).to(self.device)
        elif isinstance(clip_tensor, torch.Tensor):
            clip_tensor = clip_tensor.to(self.device)
        else:
            raise Exception(f"Given type of clip is wrong, should be torch.tensor or np.ndarray")

        # exp_tensor: 1 x T x H x W
        if isinstance(exp_tensor, np.ndarray):
            exp_tensor = torch.from_numpy(exp_tensor).to(self.device)
        elif isinstance(exp_tensor, torch.Tensor):
            exp_tensor = exp_tensor.to(self.device)
        else:
            raise Exception(f"Given type of explanation is wrong, should be torch.tensor or np.ndarray")

        # superpixel_mask: T x H x W
        if isinstance(superpixel_mask, np.ndarray):
            superpixel_mask = torch.from_numpy(superpixel_mask).to(self.device)
        elif isinstance(superpixel_mask, torch.Tensor):
            superpixel_mask = superpixel_mask.to(self.device)
        else:
            raise Exception(f"Given type of superpixel_mask is wrong, should be torch.tensor or np.ndarray")

        nc, nt, nrow, ncol = clip_tensor.shape
        assert list(exp_tensor.shape) == [1, nt, nrow, ncol]
        assert nrow == ncol
        volume = nt * nrow**2

        if mask_smooth_sigma != 0:
            exp_tensor = gaussian_blur(exp_tensor, mask_smooth_sigma)   # N x 1 x T x H x W

        if remove_method == "fade":
            base = torch.zeros_like(clip_tensor).to(self.device)    # C x T x H x W
        elif remove_method == "blur":
            base = blur(clip_tensor, klen=11, ksig=5, device=self.device)    # C x T x H x W

        exp_tensor = exp_tensor.transpose(0,1).contiguous() # T x 1 x H x W

        seg_imp_dict = {}
        for t in range(nt):
            seg_ids = torch.unique(superpixel_mask[t]).tolist()
            # print(f'Step {t}: {len(seg_ids)}, {max(seg_ids)}, {min(seg_ids)}')
            for seg_id in seg_ids:
                seg_id_imp = exp_tensor[t][0][superpixel_mask[t]==seg_id].sum().item()
                seg_imp_dict[(t, seg_id)] = seg_id_imp
        if order == "most_first":
            seg_imp_dict = {k: v for k, v in sorted(seg_imp_dict.items(), key=lambda item: item[1], reverse=True)}  # Descending
        elif order == "least_first":
            seg_imp_dict = {k: v for k, v in sorted(seg_imp_dict.items(), key=lambda item: item[1], reverse=False)}  # Ascending
        ordered_seg = list(seg_imp_dict.keys())
        n_seg = len(ordered_seg)
        # print(f'#Segments: {n_seg}')

        assert n_seg > n_step, 'Given number of steps is larger than the number of segment in superpixel masks.'

        if mode == "del":
            pmask = torch.ones(nt, nrow, ncol).to(self.device)
        elif mode == "ins":
            pmask = torch.zeros(nt, nrow, ncol).to(self.device)

        prob_lst = []
        max_prob = 0
        max_mask = None
        if vis_process:
            video_vis_list = []
            prob_vis_list = []
        
        paral_pmasks = []
        step_pts = np.linspace(0, n_seg, n_step+1).astype(np.int).tolist()
        # print(f'#Steps: {len(step_pts)}')
        for t in range(n_step+1):
            if t == 0:
                pmask = pmask
            else:
                st = step_pts[t-1]
                ed = step_pts[t]
                # print(f"Step {t}: {st}~{ed}")
                sltd_seg = ordered_seg[st:ed]
                for tidx, seg_id in sltd_seg:
                    if mode == 'del':
                        pmask[tidx][superpixel_mask[tidx]==seg_id] = 0
                    elif mode == 'ins':
                        pmask[tidx][superpixel_mask[tidx]==seg_id] = 1
            paral_pmasks.append(pmask.clone())

            if (t+1) % parallel_size == 0 or t == n_step:
                paral_pmasks = torch.stack(paral_pmasks, dim=0) # P x T x H x W
                paral_pmasks = paral_pmasks.unsqueeze(1)        # P x 1 x T x H x W

                paral_size = paral_pmasks.shape[0]
                paral_clip_tensor = clip_tensor.unsqueeze(0).repeat_interleave(paral_size, dim=0) # P x C x T x H x W
                paral_base = base.unsqueeze(0).repeat_interleave(paral_size, dim=0) # P x C x T x H x W
                paral_perturb_clip = paral_clip_tensor * paral_pmasks + paral_base * (1 - paral_pmasks) # P x C x T x H x W
                
                paral_perturb_clip.requires_grad_(False)
                paral_y = self.model(paral_perturb_clip)    # P * num_classes

                label = torch.tensor([class_id, ]*paral_size)
                prob, pred_label, pred_label_prob = process_activations(paral_y, label, softmaxed=True)

                if vis_process:
                    for p_idx in range(paral_size):
                        perturb_clip_cpu = paral_perturb_clip[p_idx].detach().cpu()
                        mask_cpu = paral_pmasks[p_idx].detach().cpu()
                        prob_array = np.array(prob_lst + prob.cpu().tolist()[:p_idx+1])

                        video_vis_np = plot_voxel(perturb_clip_cpu, mask_cpu, return_img=True, dpi=150, 
                                            title=f"{video_name} #{len(prob_array)-1}/{n_step} P={prob_array[-1]:.3f}")
                        video_vis_list.append(video_vis_np)
                        prob_vis_np = plot_causal_metric_curve(prob_array, all_steps=n_step, return_img=True, dpi=150)
                        prob_vis_list.append(prob_vis_np)

                        mask_area = mask_cpu.sum().item() / volume
                        print(video_vis_np.shape, prob_vis_np.shape, f"Acc={prob_array[-1]:.3f}", f"S={mask_area:.3f}")

                prob_lst += prob.cpu().tolist()
                paral_pmasks = []

        if vis_process:
            fig, axs = plt.subplots(1, 2, figsize=(22, 8), gridspec_kw={'width_ratios': [2, 1]}, 
                                            constrained_layout=True)
            ani = animation.FuncAnimation(fig, ani_update, fargs=(video_vis_list, prob_vis_list, axs), 
                                            interval=1000, frames=len(video_vis_list))
            if video_name != None and vis_dir != None:
                video_name = video_name.split('/')[-1]
                ani.save(f'{vis_dir}/{video_name}.gif', writer="imagemagick")
                
        prob_lst = torch.tensor(prob_lst)    # T*H'*W'
        return prob_lst