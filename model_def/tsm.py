import torch
from torch import nn

import os
import sys
sys.path.append(".")
sys.path.append("..")

class tsm (nn.Module):
    def __init__ (self, num_classes, segment_count, pretrained, with_softmax=False):
        super(tsm, self).__init__()
        self.pretrained = pretrained
        self.with_softmax = with_softmax
        self.repo = 'epic-kitchens/action-models'
        if 'epic-kitchens' in self.pretrained:
            all_classes_num = (125, 352)
            self.model = torch.hub.load(self.repo, 'TSM', all_classes_num, segment_count, 'RGB',
                                            base_model='resnet50', pretrained='epic-kitchens')
        elif 'kinetics' in self.pretrained:
            kinetics_classes_num = 400
            self.model = torch.hub.load(self.repo, 'TSM', kinetics_classes_num, segment_count, 'RGB',
                                            base_model='resnet50')
            checkpoint_path = 'model_param/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth'
            assert os.path.isfile(checkpoint_path), \
                    f'Something wrong with pretrained parameters of TSM-Kinetics, Given {checkpoint_path}.'
            print(f'Load checkpoint of TSM from {checkpoint_path}.')
            state_dict = torch.load(checkpoint_path)['state_dict']
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        elif 'sthsthv2' in self.pretrained:
            sthsthv2_num_classes = 174
            self.model = torch.hub.load(self.repo, 'TSM', sthsthv2_num_classes, segment_count, 'RGB',
                                            base_model='resnet50')
            checkpoint_path = f'model_param/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment{segment_count}_e45.pth'
            assert os.path.isfile(checkpoint_path), \
                    f'Something wrong with pretrained parameters of TSM-Kinetics, Given {checkpoint_path}.'
            print(f'Load checkpoint of TSM from {checkpoint_path}.')
            state_dict = torch.load(checkpoint_path)['state_dict']
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)

    def forward (self, inp):
        if 'epic-kitchens' in self.pretrained:
            feat = self.model.features(inp)
            verb_logits, noun_logits = self.model.logits(feat)
            if 'noun' in self.pretrained:
                logits = noun_logits
            elif 'verb' in self.pretrained:
                logits = verb_logits
        elif 'kinetics' in self.pretrained:
            logits = self.model(inp)
        elif 'sthsthv2' in self.pretrained:
            logits = self.model(inp)

        if self.with_softmax:
            probs = nn.functional.softmax(logits, dim=1)
            return probs
        else:
            return logits