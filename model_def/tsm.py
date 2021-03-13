import torch
from torch import nn

import os
import sys
sys.path.append(".")
sys.path.append("..")

from utils.CalAcc import AverageMeter, accuracy
from utils.CalAcc import process_activations
from utils.TSN import TSN

class tsm (nn.Module):
    def __init__ (self, num_classes, segment_count, pretrained, with_softmax=False):
        super(tsm, self).__init__()
        self.pretrained = pretrained
        self.with_softmax = with_softmax
        # self.repo = 'epic-kitchens/action-models'
        # if 'epic-kitchens' in self.pretrained:
        #     all_classes_num = (125, 352)
        #     self.model = torch.hub.load(self.repo, 'TSM', all_classes_num, segment_count, 'RGB',
        #                                     base_model='resnet50', pretrained='epic-kitchens')
        # elif 'kinetics' in self.pretrained:
        #     kinetics_classes_num = 400
        #     self.model = torch.hub.load(self.repo, 'TSM', kinetics_classes_num, segment_count, 'RGB',
        #                                     base_model='resnet50')
        #     checkpoint_path = 'model_param/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth'
        #     assert os.path.isfile(checkpoint_path), \
        #             f'Something wrong with pretrained parameters of TSM-Kinetics, Given {checkpoint_path}.'
        #     print(f'Load checkpoint of TSM from {checkpoint_path}.')
        #     state_dict = torch.load(checkpoint_path)['state_dict']
        #     state_dict = {k[7:]: v for k, v in state_dict.items()}
        #     self.model.load_state_dict(state_dict)
        # elif 'sthsthv2' in self.pretrained:
        #     sthsthv2_num_classes = 174
        #     self.model = torch.hub.load(self.repo, 'TSM', sthsthv2_num_classes, segment_count, 'RGB',
        #                                     base_model='resnet50')
        #     checkpoint_path = f'model_param/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment{segment_count}_e45.pth'
        #     assert os.path.isfile(checkpoint_path), \
        #             f'Something wrong with pretrained parameters of TSM-Kinetics, Given {checkpoint_path}.'
        #     print(f'Load checkpoint of TSM from {checkpoint_path}.')
        #     state_dict = torch.load(checkpoint_path)['state_dict']
        #     # print(list(state_dict.keys()))
        #     # print(list(self.model.state_dict().keys()))
        #     # print(self.model.features)
        #     state_dict = {k[7:]: v for k, v in state_dict.items()}
        #     self.model.load_state_dict(state_dict)
        self.model = TSN(num_classes, segment_count, 'RGB', base_model='resnet50',
                        consensus_type='avg', img_feature_dim=256, pretrain='imagenet',
                        is_shift=True, shift_div=8, shift_place='blockres',
                        non_local=False)

        if 'sthsthv2' in self.pretrained:
            checkpoint_path = f'model_param/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment{segment_count}_e45.pth'
            checkpoint = torch.load(checkpoint_path)
            checkpoint = checkpoint['state_dict']

            base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
            replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                            'base_model.classifier.bias': 'new_fc.bias'}
            for k, v in replace_dict.items():
                if k in base_dict:
                    base_dict[v] = base_dict.pop(k)

            self.model.load_state_dict(base_dict)

    def forward (self, inp):
        bs, ch, nt, h, w = inp.shape
        # reshaped_inp = inp.transpose(1,2).reshape(bs, -1, h, w)
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
            data_in = inp.transpose(1,2).contiguous()   # NxTxCxHxW
            # data_in = inp
            logits = self.model(data_in)
            # print(logits.shape)

        if self.with_softmax:
            probs = nn.functional.softmax(logits, dim=1)
            return probs
        else:
            return logits

    def to_device(self, device):
        self.device = device
        self.model.to(device)

    def val_model (self, dataloader, checkpoint_name=None):
        # load best model weights
        if checkpoint_name != None:
            self.load_weights(checkpoint_name)
        self.model.eval()   # Set model to evaluate mode

        y_pred = []
        y_true = []

        # running_corrects = 0
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # Iterate over data.
        # for samples in tqdm(dataloader):
        video_pred_dict = {}
        for samples in dataloader:
            inputs = samples[0].to(self.device)
            labels = samples[1].to(self.device)
            labels = labels.to(dtype=torch.long)

            video_id = samples[2]
            action_name = samples[4]

            # forward
            with torch.set_grad_enabled(False):
                # Get model outputs and calculate loss
                outputs = self.forward(inputs)
                _, preds = torch.max(outputs, 1)

            # calculating comfusion matrix
            y_pred.append(preds.detach().cpu())
            y_true.append(labels.data.detach().cpu())

            # statistics
            acc1, acc5 = accuracy(outputs, labels.data, topk=(1,5))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            
            bs = inputs.shape[0]
            probs = process_activations(outputs, labels, softmaxed=False)[0]
            for bidx in range(bs):
                video_name = samples[2][bidx].split("/")[-1]
                label = labels.data[bidx]
                video_pred_dict[video_name] = probs.detach().cpu()[bidx].item()
                # print(f"{video_name}: {video_pred_dict[video_name]}")

        video_pred_dict = {name: pred for name, pred in sorted(video_pred_dict.items(), key=lambda item: item[1], reverse=True)}
        # for video_name, pred in video_pred_dict.items():
        #     print(f"{video_name}: {pred:.3f}")

        print(' Val: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))
        print()

        y_pred = torch.cat(y_pred, dim=0).numpy()
        y_true = torch.cat(y_true, dim=0).numpy()

        return y_pred, y_true