import torch
import numpy as np 

def integrated_grad_show (grads, polarity):
    positive_channel = [0, 1, 0]  # Green
    negative_channel = [1, 0, 0]  # Red
    assert polarity in ['both', 'positive', 'negative']

    if polarity == 'both':
        grads = torch.abs(grads)
        channel = torch.tensor([1,1,1]).reshape(1,3,1,1,1)
    elif polarity == 'positive':
        grads = torch.clamp(grads, min=0.0)
        channel = torch.tensor(positive_channel).reshape(1,3,1,1,1)
    elif polarity == 'negative':
        grads = -1.0 * torch.clamp(grads, max=0.0)
        channel = torch.tensor(negative_channel).reshape(1,3,1,1,1)
    grads = grads.mean(dim=1, keepdim=True)  # convert to gray, Nx1xTxHxW

    grads_np = grads.cpu().numpy()
    grads_np = np.reshape(grads_np, (bs, -1))
    vmax = np.percentile(grads_np, 99.9, axis=1)
    vmin = np.min(grads_np, axis=1)
    normed_grads = torch.from_numpy(np.clip((grads_np - vmin) / (vmax - vmin), 0, 1))    # Nx1xTxHxW
    normed_grads = np.reshape(normed_grads, (bs, 1, nt, h, w))
    grad_show = normed_grads.repeat_interleave(3, dim=1) * channel    # Nx3xTxHxW
    return normed_grads, grad_show

def integrated_grad (inputs, labels, model, device, steps, 
                        polarity='both', show_gray=True):
    model.eval()   # Set model to evaluate mode

    bs, ch, nt, h, w = inputs.shape
    assert ch == 3
    assert labels.shape[0] == bs

    inputs = inputs.to(device)
    # labels = labels.to(device)
    labels = labels.to(dtype=torch.long)

    baseline = torch.zeros_like(inputs, device=device)
    baseline[:, 0, ...] = -1.8952
    baseline[:, 1, ...] = -1.7822
    baseline[:, 2, ...] = -1.7349
    baseline_out = model(baseline)

    backward_signals = torch.zeros_like(baseline_out, device=device)
    for bidx in range(bs):
        backward_signals[bidx, labels[bidx].cpu().item()] = 1.0

    intg_grads = 0
    for i in range(steps):
        scaled_inputs = baseline + (float(i) / steps) * (inputs - baseline)
        scaled_inputs.requires_grad = True

        # Forward
        outputs = model(scaled_inputs)
        _, preds = torch.max(outputs, dim=1)

        # Backward
        outputs.backward(backward_signals)
        intg_grads += scaled_inputs.grad.cpu() / steps
    intg_grads *= (inputs - baseline).detach().cpu()
    # normed_grads, grad_show = integrated_grad_show(intg_grads, polarity)
    if polarity == 'both':
        grads = torch.abs(intg_grads)
    elif polarity == 'positive':
        grads = torch.clamp(intg_grads, min=0.0)
    elif polarity == 'negative':
        grads = -1.0 * torch.clamp(intg_grads, max=0.0)
    grads = grads.mean(dim=1, keepdim=True)  # convert to gray, Nx1xTxHxW

    grads_np = grads.cpu().numpy()
    grads_np = np.reshape(grads_np, (bs, -1))
    vmax = np.percentile(grads_np, 99.9, axis=1, keepdims=True)
    vmin = np.min(grads_np, axis=1, keepdims=True)
    normed_grads = torch.from_numpy(np.clip((grads_np - vmin) / (vmax - vmin), 0, 1))    # Nx1xTxHxW
    normed_grads = np.reshape(normed_grads, (bs, 1, nt, h, w))
    return normed_grads

