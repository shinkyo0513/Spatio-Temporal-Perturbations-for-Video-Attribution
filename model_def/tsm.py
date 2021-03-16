import torch
from torch import nn

import os
import sys
sys.path.append(".")
sys.path.append("..")

import time
import copy
from tqdm import tqdm
import numpy as np

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

            num_ftrs = self.model.new_fc.in_features
            self.model.new_fc = nn.Linear(num_ftrs, 174)
            self.model.load_state_dict(base_dict)
            self.model.new_fc = nn.Linear(num_ftrs, num_classes)

    def load_weights(self, weights_dir):
        model_wts = torch.load(weights_dir, map_location=torch.device('cpu'))
        # parallel_wts = ("module" in list(model_wts.keys())[0])
        # if self.parallel==True and parallel_wts!=True:
        #     model_wts = {"module."+name: model_wts[name] for name in model_wts.keys()}
        # if self.parallel!=True and parallel_wts==True:
        #     model_wts = {name[7:]: model_wts[name] for name in model_wts.keys()}
        self.model.load_state_dict(model_wts)

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

    def train_model(self, dataloaders, criterion, optimizer, checkpoint_name, num_epochs, scheduler=None):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for samples in tqdm(dataloaders[phase]):
                # for samples in dataloaders[phase]:
                    inputs = samples[0].to(self.device)
                    labels = samples[1].to(self.device, dtype=torch.long)
                    # print(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # with amp.autocast(enabled=use_amp):
                        # Get model outputs and calculate loss
                        outputs = self.forward(inputs)    # assume outputs are softmax activations
                        if self.with_softmax:
                            loss = criterion(torch.log(outputs), labels)    # Using NLLLoss
                        else:
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            if scheduler != None:
                scheduler.step()
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        torch.save(best_model_wts, checkpoint_name)
        print(f"Saved the weights of the best epoch in {checkpoint_name}")
        return val_acc_history

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