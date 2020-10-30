import torch
import torch.nn.functional as F
import numpy as np
import gc

def grad_cam_3d (inputs, labels, model, device, layer_name, norm_vis=True):
    model.eval()   # Set model to evaluate mode
    
    bs, ch, nt, h, w = inputs.shape
    assert ch == 3
    assert labels.shape[0] == bs

    # Backward hook
    backward_hook = lambda li: lambda m, i_grad, o_grad: li.insert(0, o_grad[0].detach())
    # Forward hook
    forward_hook = lambda li: lambda m, i, o: li.append(o.detach())

    # print(model)

    observ_layer = model
    for name in layer_name:
        observ_layer = dict(observ_layer.named_children())[name]
    # print(observ_layer)

    observ_grad_ = []
    observ_layer.register_backward_hook(backward_hook(observ_grad_))
    observ_actv_ = []
    observ_layer.register_forward_hook(forward_hook(observ_actv_))

    inputs = inputs.to(device)
    labels = labels.to(dtype=torch.long)

    # Forward pass
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    observ_actv = observ_actv_[0]   # N x C x num_f/8 x 56 x 56
    # print('observ_actv:', observ_actv.shape)
    observ_actv = torch.repeat_interleave(observ_actv, int(nt/observ_actv.shape[2]), dim=2)

    # backward pass
    backward_signals = torch.zeros_like(outputs, device=device)
    for bidx in range(bs):
        backward_signals[bidx, labels[bidx].cpu().item()] = 1.0
    outputs.backward(backward_signals)

    observ_grad = observ_grad_[0]   # N x C x num_f/8 x 56 x 56
    # print('observ_grad:', observ_grad.shape)
    observ_grad = torch.repeat_interleave(observ_grad, int(nt/observ_grad.shape[2]), dim=2)

    observ_grad_w = observ_grad.mean(dim=4, keepdim=True).mean(dim=3, keepdim=True) # N x 512 x num_f x 1x1
    out_masks = F.relu( (observ_grad_w*observ_actv).sum(dim=1, keepdim=True) ) # N x 1 x num_f x 14x14
    out_masks = out_masks.detach().cpu()

    if norm_vis:
        normed_masks = out_masks.view(bs, -1)
        mins = torch.min(normed_masks, dim=1, keepdim=True)[0]
        maxs = torch.max(normed_masks, dim=1, keepdim=True)[0]
        normed_masks = (normed_masks - mins) / (maxs - mins)    
        out_masks = normed_masks.reshape(out_masks.shape)

    # print(out_masks.shape)

    return out_masks

def grad_cam_rnn (inputs, labels, model, device, layer_name, norm_vis=True):
    model.eval()   # Set model to evaluate mode
    
    bs, ch, nt, h, w = inputs.shape
    assert ch == 3
    assert labels.shape[0] == bs

    # Backward hook
    backward_hook = lambda li: lambda m, i_grad, o_grad: li.insert(0, o_grad[0].detach()) # layer7: 4x1024x7x7
    # backward_hook = lambda li: lambda m, i_grad, o_grad: li.insert(0, o_grad.detach())
    # Forward hook
    forward_hook = lambda li: lambda m, i, o: li.append(o.detach()) # layer7: 4x1024x7x7
    # forward_hook = lambda li: lambda m, i, o: li.append(o.detach())
    observ_layer = model
    for name in layer_name:
        # print(dict(observ_layer.named_children()).keys())
        observ_layer = dict(observ_layer.named_children())[name]

    # print(observ_layer)

    observ_grad_ = []
    bh = observ_layer.register_backward_hook(backward_hook(observ_grad_))
    # observ_layer.register_backward_hook(backward_hook)
    observ_actv_ = []
    fh = observ_layer.register_forward_hook(forward_hook(observ_actv_))
    # observ_layer.register_forward_hook(forward_hook)

    inputs = inputs.to(device)
    labels = labels.to(dtype=torch.long)

    # Forward pass
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    observ_actv = torch.stack(observ_actv_, dim=2)   # N x 512 x num_f x14x14
    # print(f'actv: min: {observ_actv.min()}, max: {observ_actv.max()}')

    # backward pass
    backward_signals = torch.zeros_like(outputs, device=device)
    for bidx in range(bs):
        backward_signals[bidx, labels[bidx].cpu().item()] = 1.0
    outputs.backward(backward_signals)

    observ_grad = torch.stack(observ_grad_, dim=2)   # N x 512 x num_f x14x14
    # print(f'grad: {observ_grad.shape}; actv: {observ_actv.shape}')
    # print(f'actv: min: {observ_grad.min()}, max: {observ_grad.max()}')

    observ_grad_w = observ_grad.mean(dim=4, keepdim=True).mean(dim=3, keepdim=True) # N x 512 x num_f x1x1
    out_masks = F.relu( (observ_grad_w*observ_actv).sum(dim=1, keepdim=True) ) # N x 1 x num_f x14x14
    out_masks = out_masks.detach().cpu()
    # print(f'min: {out_masks.min()}, max: {out_masks.max()}')

    fh.remove()
    bh.remove()

    if norm_vis:
        normed_masks = out_masks.view(bs, -1)
        mins = torch.min(normed_masks, dim=1, keepdim=True)[0]
        maxs = torch.max(normed_masks, dim=1, keepdim=True)[0]
        normed_masks = (normed_masks - mins) / (maxs - mins)    
        out_masks = normed_masks.reshape(out_masks.shape)

    return out_masks

def grad_cam (inputs, labels, model, model_name, device, layer_name, norm_vis=True):
    if model_name == 'r2p1d':
        layer_name = ['module', 'model'] + layer_name
        return grad_cam_3d(inputs, labels, model, device, layer_name, norm_vis)
    elif model_name == 'v16l':
        layer_name = ['module', 'model'] + layer_name
        return grad_cam_rnn(inputs, labels, model, device, layer_name, norm_vis)
    elif model_name == 'r50l':
        layer_name = ['module', 'model', 'resnet'] + layer_name
        return grad_cam_rnn(inputs, labels, model, device, layer_name, norm_vis)
    else:
        raise Exception(f'Unsupoorted class: given {model_name}')