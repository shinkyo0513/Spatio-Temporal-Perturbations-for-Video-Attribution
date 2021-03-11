import torch
import numpy as np 
from torchray.attribution.excitation_backprop import excitation_backprop, update_resnet
from torchray.attribution.excitation_backprop import ExcitationBackpropContext


def excitation_bp_2d (inputs, labels, model, device, layer_name, norm_vis=True):
    model.eval()   # Set model to evaluate mode
    # print(model)
    # print(dict(model.named_modules()).keys())

    observ_layer = model
    for name in layer_name:
        # print(dict(observ_layer.named_children()).keys())
        observ_layer = dict(observ_layer.named_children())[name]
    # print(observ_layer)

    bs0, nch0, nt0, h0, w0 = inputs.shape

    eb_grads = excitation_backprop(model, inputs, labels, saliency_layer=observ_layer)  # N*Tx1xHxW
    eb_grads = torch.stack(torch.split(eb_grads, nt0, dim=0), dim=0).transpose(1,2) # N x 1 x numf x 14x14
    eb_grads = torch.abs(eb_grads)  # N*Tx1xHxW
    # eb_grads = torch.clamp(eb_grads, min=0.0)
    eb_grads = eb_grads.mean(dim=1, keepdim=True)  # Nx1xTxHxW
    bs, nch, nt, h, w = eb_grads.shape

    if norm_vis:
        grads_np = eb_grads.cpu().numpy()
        grads_np = np.reshape(grads_np, (bs, -1))
        vmax = np.percentile(grads_np, 99.0, axis=1, keepdims=True)
        vmin = np.percentile(grads_np, 0.0, axis=1, keepdims=True)
        out_masks = torch.from_numpy(np.clip((grads_np - vmin) / (vmax - vmin), 0, 1))    # Nx1xTxHxW
        out_masks = out_masks.reshape(bs, 1, nt, h, w)
    else:
        out_masks = eb_grads

    if nt != nt0:
        out_masks = torch.repeat_interleave(out_masks, int(nt0/nt), dim=2)
    
    return out_masks

def excitation_bp_3d (inputs, labels, model, device, layer_name, norm_vis=True):
    model.eval()   # Set model to evaluate mode
    # print(model)
    # print(dict(model.named_modules()).keys())

    observ_layer = model
    for name in layer_name:
        # print(dict(observ_layer.named_children()).keys())
        observ_layer = dict(observ_layer.named_children())[name]
    # print(observ_layer)

    bs0, nch0, nt0, h0, w0 = inputs.shape

    eb_grads = excitation_backprop(model, inputs, labels, saliency_layer=observ_layer)  # Nx3xTxHxW
    eb_grads = torch.abs(eb_grads)  # Nx3xTxHxW
    # eb_grads = torch.clamp(eb_grads, min=0.0)
    eb_grads = eb_grads.mean(dim=1, keepdim=True)  # Nx1xTxHxW
    bs, nch, nt, h, w = eb_grads.shape

    if norm_vis:
        grads_np = eb_grads.cpu().numpy()
        grads_np = np.reshape(grads_np, (bs, -1))
        vmax = np.percentile(grads_np, 99.0, axis=1, keepdims=True)
        vmin = np.percentile(grads_np, 0.0, axis=1, keepdims=True)
        out_masks = torch.from_numpy(np.clip((grads_np - vmin) / (vmax - vmin), 0, 1))    # Nx1xTxHxW
        out_masks = out_masks.reshape(bs, 1, nt, h, w)
    else:
        out_masks = eb_grads

    if nt != nt0:
        out_masks = torch.repeat_interleave(out_masks, int(nt0/nt), dim=2)
    
    return out_masks

def excitation_bp_rnn (inputs, labels, model, device, layer_name, norm_vis=True):
    model.eval()   # Set model to evaluate mode
    # print(model)
    model.module.model.resnet = update_resnet(model.module.model.resnet, debug=False)
    # model.model.resnet = update_resnet(model.model.resnet, debug=True)
    
    bs, ch, nt, h, w = inputs.shape
    assert ch == 3
    assert labels.shape[0] == bs

    inputs = inputs.to(device)
    labels = labels.to(dtype=torch.long)
    inputs.requires_grad_()

    # Backward hook
    backward_hook = lambda li: lambda m, i_grad, o_grad: li.insert(0, o_grad[0]) # layer7: 4x1024x7x7
    # Forward hook
    forward_hook = lambda li: lambda m, i, o: li.append(o) # layer7: 4x1024x7x7

    observ_layer = model
    for name in layer_name:
        observ_layer = dict(observ_layer.named_children())[name]

    mid_layer_name = layer_name[:-2] + ["fc7"]
    mid_layer = model
    for name in mid_layer_name:
        mid_layer = dict(mid_layer.named_children())[name]

    mid_actvs = []
    mfh = mid_layer.register_forward_hook(forward_hook(mid_actvs))

    observ_actvs = []
    ofh = observ_layer.register_forward_hook(forward_hook(observ_actvs))

    with ExcitationBackpropContext():
        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # backward pass
        backward_signals = torch.zeros_like(outputs, device=device)
        for bidx in range(bs):
            backward_signals[bidx, labels[bidx].item()] = 1.0

        mid_grads = []
        for fidx, mid_actv in enumerate(mid_actvs):
            mid_grad = torch.autograd.grad(outputs, mid_actv, grad_outputs=backward_signals, 
                                            retain_graph=True, allow_unused=True)[0]
            mid_grads.append(mid_grad)
        mid_grads = torch.stack(mid_grads, dim=-1)  # N x 512 x 16
        # print(f'mid_grads: {mid_grads.shape}')
        # print(f'mid_grads: min: {mid_grads.min()}, max: {mid_grads.max()}')

        normed_mid_grads = nt * mid_grads / mid_grads.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
        # print(f'normed_mid_grads: {normed_mid_grads.shape}')
        # print(f'normed_mid_grads: min: {normed_mid_grads.min()}, max: {normed_mid_grads.max()}')

        normed_mid_grads = torch.unbind(normed_mid_grads, dim=-1)
        observ_grads = []
        for fidx, normed_mid_grad in enumerate(normed_mid_grads):
            observ_grad = torch.autograd.grad(mid_actvs[fidx], 
                                                observ_actvs[fidx], 
                                                grad_outputs=normed_mid_grad,
                                                retain_graph=True, allow_unused=True)[0]
            observ_grads.append(observ_grad)
        observ_grads = torch.stack(observ_grads, dim=2)     # N x 512 x num_f x7x7
        # print(f'observ_grads: {observ_grads.shape}')
        # print(f'observ_grads: min: {observ_grads.min()}, max: {observ_grads.max()}')

    out_masks = torch.abs(observ_grads)
    out_masks = out_masks.mean(dim=1, keepdim=True)   # N x 1 x num_f x7x7
    out_masks = out_masks.detach().cpu()
    # out_masks = torch.nn.functional.relu(out_masks)

    ofh.remove()
    mfh.remove()

    if norm_vis:
        normed_masks = out_masks.view(bs, -1)
        mins = torch.min(normed_masks, dim=1, keepdim=True)[0]
        maxs = torch.max(normed_masks, dim=1, keepdim=True)[0]
        normed_masks = (normed_masks - mins) / (maxs - mins)    
        out_masks = normed_masks.reshape(out_masks.shape)

    return out_masks

def excitation_bp (inputs, labels, model, model_name, device, layer_name, norm_vis=True):
    if model_name == 'r2p1d':
        layer_name = ['module', 'model'] + layer_name
        return excitation_bp_3d(inputs, labels, model, device, layer_name, norm_vis)
    elif model_name == 'r50l':
        layer_name = ['module', 'model', 'resnet'] + layer_name
        return excitation_bp_rnn(inputs, labels, model, device, layer_name, norm_vis)
    if model_name == 'tsm':
        layer_name = ['module', 'model', 'base_model'] + layer_name
        return excitation_bp_2d(inputs, labels, model, device, layer_name, norm_vis)
    else:
        raise Exception(f'Unsupoorted class: given {model_name}')
