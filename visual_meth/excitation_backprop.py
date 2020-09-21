import torch
import numpy as np 
from torchray.attribution.excitation_backprop import excitation_backprop

def excitation_bp (inputs, labels, model):
    model.eval()   # Set model to evaluate mode

    bs, nch, nt, h, w = inputs.shape

    eb_grads = excitation_backprop(model, inputs, labels)  # Nx3xTxHxW
    # eb_grads = torch.abs(eb_grads)
    eb_grads = torch.clamp(eb_grads, min=0.0)
    eb_grads = eb_grads.mean(dim=1, keepdim=True)  # Nx1xTxHxW

    grads_np = eb_grads.cpu().numpy()
    print(grads_np.shape)
    print(grads_np.min(), grads_np.max(), grads_np.sum()/(bs*nt*h*w))

    grads_np = np.reshape(grads_np, (bs, -1))
    vmax = np.percentile(grads_np, 99.0, axis=1, keepdims=True)
    vmin = np.percentile(grads_np, 0.0, axis=1, keepdims=True)
    normed_grads = torch.from_numpy(np.clip((grads_np - vmin) / (vmax - vmin), 0, 1))    # Nx1xTxHxW
    normed_grads = normed_grads.reshape(bs, 1, nt, h, w)
    return normed_grads
