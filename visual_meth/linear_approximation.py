import torch
import numpy as np 
from torchray.attribution.linear_approx import linear_approx

def linear_appr (inputs, labels, model):
    model.eval()   # Set model to evaluate mode
    # print(dict(model.named_modules()).keys())
    bs0, nch0, nt0, h0, w0 = inputs.shape

    eb_grads = linear_approx(model, inputs, labels)
    eb_grads = torch.abs(eb_grads)  # Nx3xTxHxW
    # eb_grads = torch.clamp(eb_grads, min=0.0)
    eb_grads = eb_grads.mean(dim=1, keepdim=True)  # Nx1xTxHxW

    grads_np = eb_grads.detach().cpu().numpy()
    # print(grads_np.shape)
    bs, nch, nt, h, w = grads_np.shape
    # print(grads_np.min(), grads_np.max(), grads_np.sum()/(bs*nt*h*w))

    grads_np = np.reshape(grads_np, (bs, -1))
    vmax = np.percentile(grads_np, 99.0, axis=1, keepdims=True)
    vmin = np.percentile(grads_np, 0.0, axis=1, keepdims=True)
    normed_grads = torch.from_numpy(np.clip((grads_np - vmin) / (vmax - vmin), 0, 1))    # Nx1xTxHxW
    normed_grads = normed_grads.reshape(bs, 1, nt, h, w)
    if nt != nt0:
        normed_grads = torch.repeat_interleave(normed_grads, int(nt0/nt), dim=2)
    return normed_grads
