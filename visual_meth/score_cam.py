import torch
import torch.nn.functional as F
import numpy as np
import gc

def get_activations_2d (inputs, labels, model, device, layer_name, norm_vis=True):
    model.eval()   # Set model to evaluate mode
    
    bs, ch, nt, h, w = inputs.shape
    assert ch == 3
    assert labels.shape[0] == bs

    # Forward hook
    forward_hook = lambda li: lambda m, i, o: li.append(o.detach())

    # print(model)
    observ_layer = model
    for name in layer_name:
        observ_layer = dict(observ_layer.named_children())[name]
    # print(observ_layer)

    observ_actv_ = []
    fh = observ_layer.register_forward_hook(forward_hook(observ_actv_))

    inputs = inputs.to(device)
    labels = labels.to(dtype=torch.long)

    # Forward pass
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    observ_actv = observ_actv_[0]   # N*numf x C x 56 x 56
    # print('observ_actv:', observ_actv.shape)

    fh.remove()
    return observ_actv

def get_activations_3d (inputs, labels, model, device, layer_name, norm_vis=True):
    model.eval()   # Set model to evaluate mode
    
    bs, ch, nt, h, w = inputs.shape
    assert ch == 3
    assert labels.shape[0] == bs

    # Forward hook
    forward_hook = lambda li: lambda m, i, o: li.append(o.detach())

    observ_layer = model
    for name in layer_name:
        observ_layer = dict(observ_layer.named_children())[name]
    # print(observ_layer)

    observ_actv_ = []
    fh = observ_layer.register_forward_hook(forward_hook(observ_actv_))

    inputs = inputs.to(device)
    labels = labels.to(dtype=torch.long)

    # Forward pass
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    observ_actv = observ_actv_[0]   # N x C x num_f/8 x 56 x 56
    # print('observ_actv:', observ_actv.shape)
    observ_actv = torch.repeat_interleave(observ_actv, int(nt/observ_actv.shape[2]), dim=2)

    fh.remove()
    return observ_actv

def get_activations_rnn(inputs, labels, model, device, layer_name, norm_vis=True):
    model.eval()   # Set model to evaluate mode
    
    bs, ch, nt, h, w = inputs.shape
    assert ch == 3
    assert labels.shape[0] == bs

    # Forward hook
    forward_hook = lambda li: lambda m, i, o: li.append(o.detach()) # layer7: 4x1024x7x7
    # forward_hook = lambda li: lambda m, i, o: li.append(o.detach())
    observ_layer = model
    for name in layer_name:
        # print(dict(observ_layer.named_children()).keys())
        observ_layer = dict(observ_layer.named_children())[name]

    observ_actv_ = []
    fh = observ_layer.register_forward_hook(forward_hook(observ_actv_))

    inputs = inputs.to(device)
    labels = labels.to(dtype=torch.long)

    # Forward pass
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    observ_actv = torch.stack(observ_actv_, dim=2)   # N x 512 x num_f x14x14
    # print(observ_actv.shape)
    # print(f'actv: min: {observ_actv.min()}, max: {observ_actv.max()}')
    fh.remove()
    return observ_actv

def score_cam (inputs, labels, model, model_name, device, layer_name, norm_vis=True):
    with torch.no_grad():
        if model_name == 'r2p1d':
            layer_name = ['module', 'model'] + layer_name
            activations = get_activations_3d(inputs, labels, model, device, layer_name, norm_vis)
        elif model_name == 'v16l':
            layer_name = ['module', 'model'] + layer_name
            activations = get_activations_rnn(inputs, labels, model, device, layer_name, norm_vis)
        elif model_name == 'r50l':
            layer_name = ['module', 'model', 'resnet'] + layer_name
            activations = get_activations_rnn(inputs, labels, model, device, layer_name, norm_vis)
        elif model_name == 'tsm':
            layer_name = ['module', 'model', 'base_model'] + layer_name
            # layer_name = ['module', 'model'] + layer_name
            activations = get_activations_2d(inputs, labels, model, device, layer_name, norm_vis)
        else:
            raise Exception(f'Unsupoorted class: given {model_name}')

        bs, ch, nt, h, w = inputs.shape
        out_masks = torch.zeros((bs, 1, nt, h, w), device=device)
        num_masks = activations.shape[1]

        for i in range(num_masks):
            masks = activations[:,i,...].clone()    # N x num_f x 14x14
            masks = F.interpolate(masks, size=(h, w), mode='bilinear', align_corners=False) # N x num_f x H x W

            masks = masks.view(bs, -1)
            masks_min = masks.min(dim=1, keepdim=True)[0]
            masks_max = masks.max(dim=1, keepdim=True)[0]
            # normalize to 0-1
            # if masks_min == masks_max:
            #     continue
            # print(masks_max.shape)
            norm_masks = (masks - masks_min) / (masks_max - masks_min + 1e-7)
            norm_masks = norm_masks.view(bs, 1, nt, h, w)    # N x 1 x num_f x H x W

            # how much increase if keeping the highlighted region
            # predication on masked input
            outputs = model(inputs * norm_masks)
            scores = torch.zeros_like(labels, device=device, dtype=torch.float)
            for bidx in range(bs):
                scores[bidx] = outputs[bidx][labels[bidx].item()].item()

            # print(scores.shape, norm_masks.shape, out_masks.shape)
            
            # out_masks +=  scores * norm_masks
            out_masks += torch.einsum('b,bcthw->bcthw', scores, norm_masks)

    out_masks = F.relu(out_masks)
    out_masks = out_masks.detach().cpu()

    if norm_vis:
        normed_masks = out_masks.view(bs, -1)
        mins = torch.min(normed_masks, dim=1, keepdim=True)[0]
        maxs = torch.max(normed_masks, dim=1, keepdim=True)[0]
        normed_masks = (normed_masks - mins) / (maxs - mins)    
        out_masks = normed_masks.reshape(out_masks.shape)
    
    return out_masks
