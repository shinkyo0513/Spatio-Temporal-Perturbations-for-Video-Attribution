import torch
from torch import nn
from torchvision import models

import time
import copy
from tqdm import tqdm
import numpy as np

from utils.CalAcc import AverageMeter, accuracy
from utils.CalAcc import process_activations

# from torch.cuda import amp


class r2plus1d (nn.Module):
    def __init__(self, num_classes, with_softmax=False, pretrained=True):
        super(r2plus1d, self).__init__()
        self.num_classes = num_classes

        self.model = models.video.r2plus1d_18(pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 112
        self.parallel = False
        self.with_softmax = with_softmax

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_tensor):
        pred = self.model(inp_tensor)
        if not self.with_softmax:
            return pred
        else:
            prob = self.softmax(pred)
            return prob
        # logits = self.model(inp_tensor)
        # prob = self.softmax(logits)
        # return logits, prob

    def load_weights(self, weights_dir):
        model_wts = torch.load(weights_dir, map_location=torch.device('cpu'))
        parallel_wts = ("module" in list(model_wts.keys())[0])
        if self.parallel == True and parallel_wts != True:
            model_wts = {"module."+name: model_wts[name]
                         for name in model_wts.keys()}
        if self.parallel != True and parallel_wts == True:
            model_wts = {name[7:]: model_wts[name]
                         for name in model_wts.keys()}
        self.model.load_state_dict(model_wts)

    def to_device(self, device):
        self.device = device
        self.model.to(device)

    def save_weights(self, save_dir):
        model_wts = copy.deepcopy(self.model.state_dict())
        if "module" in list(model_wts.keys())[0]:
            model_wts = {name[7:]: model_wts[name]
                         for name in model_wts.keys()}
        torch.save(model_wts, save_dir)

    def parallel_model(self, device_ids):
        print('Use', len(device_ids), 'GPUs!')
        self.model = nn.DataParallel(self.model, device_ids=device_ids)
        self.parallel = True

    def set_parameter_requires_grad(self, retrain_type):
        if retrain_type == 'OnlyFC':
            for name, param in self.model.named_parameters():
                if "fc" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif retrain_type == 'FC+FinalConv':
            for name, param in self.model.named_parameters():
                if 'layer4.1' in name or "fc" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif retrain_type == 'FC+HalfConv':
            for name, param in self.model.named_parameters():
                if 'layer4.' in name or 'layer3.' in name or "fc" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def get_parameter_to_update(self, debug=False):
        if debug:
            print("Params to learn:")
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                if debug:
                    print("\t", name)
        return params_to_update

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
                        # assume outputs are softmax activations
                        outputs = self.forward(inputs)
                        if self.with_softmax:
                            # Using NLLLoss
                            loss = criterion(torch.log(outputs), labels)
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
                epoch_acc = running_corrects.double(
                ) / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

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
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        if checkpoint_name:
            torch.save(best_model_wts, checkpoint_name)
            print(f"Saved the weights of the best epoch in {checkpoint_name}")
        return val_acc_history

    def val_model(self, dataloader, checkpoint_name=None):
        since = time.time()

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

            # forward
            with torch.set_grad_enabled(False):
                # Get model outputs and calculate loss
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

            # calculating comfusion matrix
            y_pred.append(preds.detach().cpu())
            y_true.append(labels.data.detach().cpu())

            # statistics
            acc1, acc5 = accuracy(outputs, labels.data, topk=(1, 5))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            bs = inputs.shape[0]
            probs = process_activations(outputs, labels, softmaxed=False)[0]
            for bidx in range(bs):
                video_name = samples[2][bidx].split("/")[-1]
                label = labels.data[bidx]
                video_pred_dict[video_name] = probs.detach().cpu()[bidx].item()
                # print(f"{video_name}: {video_pred_dict[video_name]}")

        video_pred_dict = {name: pred for name, pred in sorted(
            video_pred_dict.items(), key=lambda item: item[1], reverse=True)}
        # for video_name, pred in video_pred_dict.items():
        #     print(f"{video_name}: {pred:.3f}")

        # top100_dict = {}
        # for video_name in list(video_pred_dict.keys())[:100]:
        #     top100_dict[video_name] = video_pred_dict[video_name]
        # print(top100_dict)
        # torch.save(video_pred_dict, "epic_verb_r2plus1d_preds.pt")

        print(' Val: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        print()

        y_pred = torch.cat(y_pred, dim=0).numpy()
        y_true = torch.cat(y_true, dim=0).numpy()

        time_elapsed = time.time() - since
        return y_pred, y_true
