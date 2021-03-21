import torch
import numpy as np 
import math
from scipy.ndimage import gaussian_filter

import sys
sys.path.append("..")
from utils.GaussianSmoothing import GaussianSmoothing
from utils.TensorSmooth import imsmooth

def gaussian_blur(inputs, sigma):
    if sigma == 0:
        return inputs

    bs, ch, nt, h, w = inputs.shape
    reshaped_inputs = torch.cat(inputs.unbind(dim=2), dim=0)

    # kernel_size = 11
    # batch_gaussian_filter = GaussianSmoothing(channels=3, 
    #                                           kernel_size=kernel_size, 
    #                                           sigma=sigma, 
    #                                           dim=2
    #                                          ).to(inputs.device)
    # blurred_inputs = batch_gaussian_filter(reshaped_inputs)
    blurred_inputs = imsmooth(reshaped_inputs, sigma)
    blurred_inputs = torch.stack(blurred_inputs.split(bs, dim=0), dim=2)    # BxCxTxHxW
    return blurred_inputs

def blur_integrated_grad (inputs, labels, model, device, steps, 
                          grad_step=0.01, max_sigma=50, sqrt=False,
                          polarity='both'):
    model.eval()   # Set model to evaluate mode

    bs, ch, nt, h, w = inputs.shape
    assert ch == 3
    assert labels.shape[0] == bs

    inputs = inputs.to(device)
    labels = labels.to(dtype=torch.long)

    outputs = model(inputs)
    backward_signals = torch.zeros_like(outputs, device=device)
    for bidx in range(bs):
        backward_signals[bidx, labels[bidx].cpu().item()] = 1.0

    if sqrt:
        sigmas = [math.sqrt(float(i)*max_sigma/float(steps)) for i in range(0, steps+1)]
    else:
        sigmas = [float(i)*max_sigma/float(steps) for i in range(0, steps+1)]
    step_vector_diff = [sigmas[i+1] - sigmas[i] for i in range(0, steps)]

    intg_grads = 0
    for i in range(steps):
        # Process inputs
        inputs.requires_grad = True
        scaled_inputs = gaussian_blur(inputs, sigmas[i])
        # scaled_inputs.requires_grad = True
        scaled_inputs.retain_grad() 

        # Forward
        outputs = model(scaled_inputs)
        _, preds = torch.max(outputs, dim=1)

        # Backward
        outputs.backward(backward_signals)

        # Integrate Grads
        gaussian_gradient = (gaussian_blur(
            inputs, sigmas[i] + grad_step) - scaled_inputs) / grad_step
        intg_grads += step_vector_diff[i] * gaussian_gradient * scaled_inputs.grad

    # intg_grads *= -1.0

    # normed_grads, grad_show = integrated_grad_show(intg_grads, polarity)
    if polarity == 'both':
        grads = torch.abs(intg_grads)
    elif polarity == 'positive':
        grads = torch.clamp(intg_grads, min=0.0)
    elif polarity == 'negative':
        grads = -1.0 * torch.clamp(intg_grads, max=0.0)
    grads = grads.mean(dim=1, keepdim=True)  # convert to gray, Nx1xTxHxW

    grads_np = grads.cpu().detach().numpy()
    grads_np = np.reshape(grads_np, (bs, -1))
    vmax = np.percentile(grads_np, 99.9, axis=1, keepdims=True)
    vmin = np.min(grads_np, axis=1, keepdims=True)
    normed_grads = torch.from_numpy(np.clip((grads_np - vmin) / (vmax - vmin), 0, 1))    # Nx1xTxHxW
    normed_grads = normed_grads.reshape(bs, 1, nt, h, w)
    return normed_grads

