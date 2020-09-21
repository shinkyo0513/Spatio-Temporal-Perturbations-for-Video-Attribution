import torch
import numpy as np 

def gradients (inputs, labels, model, device, multiply_input=False,
                 polarity='both'):
    model.eval()   # Set model to evaluate mode

    bs, ch, nt, h, w = inputs.shape
    assert ch == 3
    assert labels.shape[0] == bs

    inputs = inputs.to(device)
    # labels = labels.to(device)
    labels = labels.to(dtype=torch.long)

    # Get model outputs and calculate loss
    inputs.requires_grad = True
    with torch.set_grad_enabled(True):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        backward_signals = torch.zeros_like(outputs, device=device)
        for bidx in range(bs):
            backward_signals[bidx, labels[bidx].cpu().item()] = 1.0
        outputs.backward(backward_signals)
    inputs_grad = inputs.grad.cpu()
    if multiply_input:
        inputs_grad *= inputs.detach().cpu()

    if polarity == 'both':
        grads = torch.abs(inputs_grad)
    elif polarity == 'positive':
        grads = torch.clamp(inputs_grad, min=0.0)
    elif polarity == 'negative':
        grads = -1.0 * torch.clamp(inputs_grad, max=0.0)
    grads = grads.mean(dim=1, keepdim=True)  # convert to gray, Nx1xTxHxW

    grads_np = grads.cpu().numpy()
    grads_np = np.reshape(grads_np, (bs, -1))
    vmax = np.percentile(grads_np, 99.9, axis=1, keepdims=True)
    vmin = np.min(grads_np, axis=1, keepdims=True)
    normed_grads = torch.from_numpy(np.clip((grads_np - vmin) / (vmax - vmin), 0, 1))    # Nx1xTxHxW
    normed_grads = normed_grads.reshape(bs, 1, nt, h, w)
    return normed_grads
