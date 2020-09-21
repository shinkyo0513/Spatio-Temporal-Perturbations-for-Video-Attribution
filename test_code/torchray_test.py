from torchray.attribution.common import Probe, get_module
from torchray.attribution.excitation_backprop import excitation_backprop
from torchray.attribution.excitation_backprop import ExcitationBackpropContext
from torchray.attribution.excitation_backprop import gradient_to_excitation_backprop_saliency
from torchray.benchmark import get_example_data, plot_example
import torch

# vgg16, x1, category_id1, _ = get_example_data(arch='vgg16')
# # print(vgg16)
# saliency1 = excitation_backprop(vgg16, x1, category_id1)
# print(saliency1.shape)

resnet18, x2, category_id2, _ = get_example_data(arch='resnet18')
# print(resnet18)
saliency2 = excitation_backprop(resnet18, x2, category_id2, saliency_layer='layer3')
print(saliency2.shape)

# from model_def.r50lstm import r50lstm
# r50l = r50lstm(num_classes=24, with_softmax=False)
# x3 = torch.randn((1,3,16,112,112), requires_grad=True)
# # print(r50l)
# saliency3 = excitation_backprop(r50l, x3, 3)
# print(saliency3.shape)

# from model_def.r2plus1d import r2plus1d
# r2p1d = r2plus1d(num_classes=24, with_softmax=True)
# x4 = torch.randn((4,3,16,112,112), requires_grad=True)
# label = torch.Tensor((1,2,3,4)).to(torch.int64)
# saliency4 = excitation_backprop(r2p1d, x4, label)
# print(saliency4.shape)