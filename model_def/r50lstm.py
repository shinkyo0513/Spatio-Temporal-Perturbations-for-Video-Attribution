import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader

import os
from os.path import isdir, isfile, join

import copy
import time
from tqdm import tqdm

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

import sys
sys.path.append(proj_root)
from utils.CalAcc import AverageMeter, accuracy
from utils.CalAcc import process_activations

class ReluLSTMCell (nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(ReluLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.fc_x = nn.Linear(input_size, 4*hidden_size, bias=bias)
        self.fc_h = nn.Linear(hidden_size, 4*hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden, cell):
        gate_input = self.fc_x(input) + self.fc_h(hidden)
        i, f, o, g = torch.split(gate_input, self.hidden_size, dim=1)
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        o = self.sigmoid(o)
        g = self.relu(g)

        new_cell = f * cell + i * g
        new_hidden = o * self.relu(new_cell)
        return new_hidden, new_cell

class Resnet50LSTM (nn.Module):
    def __init__ (self, num_classes, pretrained=True):
        super(Resnet50LSTM, self).__init__()
        self.num_classes = num_classes

        self.resnet = torchvision.models.resnet50(pretrained=False)
        if pretrained:
            pt_wgts = torch.load(os.path.join(proj_root, 'model_param', 'kinetics400_rgb_resnet18_tsn.pt'))
            model_wgts = self.resnet.state_dict()
            for k in model_wgts.keys():
                if k in pt_wgts:
                    model_wgts[k] = pt_wgts[k]
            self.resnet.load_state_dict(model_wgts)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.fc6 = nn.Linear(2048, 1024, bias=True)
        self.fc7 = nn.Linear(1024, 512, bias=True)
        # self.fc7 = nn.Linear(1024, 1024, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(True)
        self.lstm1 = ReluLSTMCell(512, 256, bias=True)
        self.fc8_final = nn.Linear(256, self.num_classes, bias=True)
        # self.lstm1 = ReluLSTMCell(1024, 512, bias=True)
        # self.fc8_final = nn.Linear(512, self.num_classes, bias=True)

    def forward (self, video_tensor):
        # video_tensor: batch_size x 3 x num_f x H x W
        batch_size = video_tensor.shape[0]
        seq_len = video_tensor.shape[2]

        logits = []
        # preds = []
        h_top = torch.zeros(batch_size, 256).to(video_tensor.device)
        c_top = torch.zeros(batch_size, 256).to(video_tensor.device)
        # h_top = torch.zeros(batch_size, 512).to(video_tensor.device)
        # c_top = torch.zeros(batch_size, 512).to(video_tensor.device)
        for fidx in range(seq_len):
            img_tensor = video_tensor[:,:,fidx,:,:]
            y = self.resnet(img_tensor)
            y = y.view(batch_size, 2048)

            y = self.dropout(self.relu(self.fc6(y)))
            y = self.dropout(self.relu(self.fc7(y)))
            h_top, c_top = self.lstm1(y, h_top.detach(), c_top.detach())
            y = self.dropout(h_top)
            y = self.fc8_final(y)

            logits.append(y)
        logits = torch.stack(logits, dim=1)   # bs x numf x num_cls
        pred = torch.mean(logits, dim=1)    # bs x num_cls
        return pred

class r50lstm (nn.Module):
    def __init__ (self, num_classes, with_softmax=False, pretrained=True):
        super(r50lstm, self).__init__()
        self.num_classes = num_classes
        self.model = Resnet50LSTM(num_classes, pretrained)

        self.parallel = False
        # if pretrained:
        #     self.load_weights(f"{proj_root}/model_param/ucf101_24_r50l_16_AllFC+LSTM_LongRange.pt")
        self.with_softmax = with_softmax
        self.softmax = nn.Softmax(dim=1)

    def forward (self, video_tensor):
        # video_tensor: batch_size x 3 x num_f x H x W
        pred = self.model(video_tensor)
        if self.with_softmax:
            prob = self.softmax(pred)
            return prob
        else:
            return pred

    def load_weights(self, weights_dir):
        model_wts = torch.load(weights_dir, map_location=torch.device('cpu'))
        parallel_wts = ("module" in list(model_wts.keys())[0])
        if self.parallel==True and parallel_wts!=True:
            model_wts = {"module."+name: model_wts[name] for name in model_wts.keys()}
        if self.parallel!=True and parallel_wts==True:
            model_wts = {name[7:]: model_wts[name] for name in model_wts.keys()}
        self.model.load_state_dict(model_wts)

    def to_device(self, device):
        self.device = device
        self.model.to(device)

    def save_weights(self, save_dir):
        model_wts = copy.deepcopy(self.model.state_dict())
        if "module" in list(model_wts.keys())[0]:
            model_wts = {name[7:]: model_wts[name] for name in model_wts.keys()}
        torch.save(model_wts, save_dir)

    def parallel_model(self, device_ids):
        print('Use', len(device_ids), 'GPUs!')
        self.model = nn.DataParallel(self.model, device_ids=device_ids)
        self.parallel = True

    def set_parameter_requires_grad(self, retrain_type):
        if retrain_type == 'FinalFC':
            for name, param in self.model.named_parameters():
                if "fc8_final" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif retrain_type == 'FinalFC+LSTM':
            for name, param in self.model.named_parameters():
                if 'lstm1' in name or "fc8_final" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif retrain_type == 'AllFC+LSTM':
            for name, param in self.model.named_parameters():
                if 'fc' in name or 'lstm1' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif retrain_type == 'FinalConv+AllFC+LSTM':
            for name, param in self.model.named_parameters():
                if 'fc' in name or 'lstm1' in name or 'conv5' in name:
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
                    print("\t",name)
        return params_to_update

    def train_model(self, dataloaders, criterion, optimizer, checkpoint_name, num_epochs, scheduler=None):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)

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
                # for samples in tqdm(dataloaders[phase]):
                for samples in dataloaders[phase]:
                    inputs = samples[0].to(self.device)
                    labels = samples[1].to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
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
        video_pred_dict = {}
        # for samples in tqdm(dataloader):
        for samples in dataloader:
            inputs = samples[0].to(self.device)
            labels = samples[1].to(self.device)
            labels = labels.to(dtype=torch.long)

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

        # top100_dict = {}
        # for video_name in list(video_pred_dict.keys())[:100]:
        #     top100_dict[video_name] = video_pred_dict[video_name]
        # print(top100_dict)
        # torch.save(video_pred_dict, "epic_verb_vgg16lstm_preds.pt")

        print(' Val: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))
        # print(top1)
        print()

        y_pred = torch.cat(y_pred, dim=0).numpy()
        y_true = torch.cat(y_true, dim=0).numpy()

        time_elapsed = time.time() - since
        return y_pred, y_true
    
if __name__ == "__main__":
    # Test the action accuracy of vgg16lstm with parameters copied from caffemodel
    from datasets.ucf101_24_dataset_new import UCF101_24_Dataset

    ds_path = f"{ds_root}/UCF101_24/"
    pt_save_dir = f"{proj_root}/model_param/ucf101_24_r50l_16_Full_LongRange.pt"
    num_f = 16
    long_range = True

    multi_gpu = True
    num_devices = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ft = r50lstm(101)
    model_ft.load_weights(pt_save_dir)
    model_ft.to_device(device)
    if multi_gpu:
        model_ft.parallel_model(device_ids=list(range(num_devices)))

    sample_mode = 'long_range_last' if long_range else "fixed"
    frame_rate = 6 if not long_range else 2
    val_dataset = UCF101_24_Dataset(ds_path, num_f, sample_mode, 1, 
                                    frame_rate, train=False, testlist_idx=2)
    print('Num of clips:{}'.format(len(val_dataset)))
    val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=True, 
                        num_workers=num_devices)
    # Validate
    y_pred, y_true = model_ft.val_model(val_dataloader)