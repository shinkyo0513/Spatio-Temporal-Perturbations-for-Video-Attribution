import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import os
from os.path import isdir, isfile, join

import copy
import time
from tqdm import tqdm

import sys
sys.path.append("/home/acb11711tx/lzq/VideoPerturb2")
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

class vgg16lstm_pytorch (nn.Module):
    def __init__ (self, num_classes):
        super(vgg16lstm_pytorch, self).__init__()
        self.num_classes = num_classes

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Linear(512*7*7, 4096, bias=True)
        self.fc7 = nn.Linear(4096, 4096, bias=True)
        self.lstm1 = ReluLSTMCell(4096, 256, bias=True)
        self.fc8_final = nn.Linear(256, self.num_classes, bias=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(True)
        self.softmax = nn.Softmax(dim=1)

    def forward (self, video_tensor):
        # video_tensor: batch_size x 3 x num_f x H x W
        batch_size = video_tensor.shape[0]
        seq_len = video_tensor.shape[2]

        probs = []
        # preds = []
        h_top = torch.zeros(batch_size, 256).to(video_tensor.device)
        c_top = torch.zeros(batch_size, 256).to(video_tensor.device)
        for fidx in range(seq_len):
            img_tensor = video_tensor[:,:,fidx,:,:]
            y = self.relu(self.conv1_1(img_tensor))
            y = self.relu(self.conv1_2(y))
            y = self.pool(y)

            y = self.relu(self.conv2_1(y))
            y = self.relu(self.conv2_2(y))
            y = self.pool(y)

            y = self.relu(self.conv3_1(y))
            y = self.relu(self.conv3_2(y))
            y = self.relu(self.conv3_3(y))
            y = self.pool(y)

            y = self.relu(self.conv4_1(y))
            y = self.relu(self.conv4_2(y))
            y = self.relu(self.conv4_3(y))
            y = self.pool(y)

            y = self.relu(self.conv5_1(y))
            y = self.relu(self.conv5_2(y))
            y = self.relu(self.conv5_3(y))
            y = self.pool5(y)
            y = y.view(batch_size, -1)

            y = self.dropout(self.relu(self.fc6(y)))
            y = self.dropout(self.relu(self.fc7(y)))
            h_top, c_top = self.lstm1(y, h_top.detach(), c_top.detach())
            y = self.dropout(h_top)
            y = self.fc8_final(y)

            prob = self.softmax(y)
            probs.append(prob)
        probs = torch.stack(probs, dim=1)   # bs x numf x num_cls
        probs = torch.mean(probs, dim=1)    # bs x num_cls
        return probs

class vgg16lstm (nn.Module):
    def __init__ (self, num_classes, pretrained=True):
        super(vgg16lstm, self).__init__()
        self.num_classes = num_classes

        self.model = vgg16lstm_pytorch(101)
        self.parallel = False

        if pretrained:
            self.load_weights("/home/acb11711tx/lzq/VideoPerturb2/models/ucf101_24_vgg16lstm.pt")
        num_ftrs = self.model.fc8_final.in_features
        self.model.fc8_final = nn.Linear(num_ftrs, num_classes)

    def forward (self, inp_tensor):
        return self.model(inp_tensor)

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

    def load_weights_from_caffemodel (self, save=False):
        import caffe
        caffe_root = '/home/acb11711tx/lzqCaffe-ExcitationBP-RNNs-master'
        model_file = caffe_root + f'/models/VGG16_LSTM/deploy.prototxt'
        model_weights = caffe_root + '/models/VGG16_LSTM/VGG16LSTM_UCF101plusBU101.caffemodel'
        caffe_net = caffe.Net(model_file, model_weights, caffe.TEST)

        torch_param_dict = {}
        for layer, params in caffe_net.params.items():
            print(layer)
            layer = layer.replace("-", "_")
            for idx, param in enumerate(params):
                if "lstm" in layer:
                    param_type = "fc_x." if idx<=1 else "fc_h."
                else:
                    param_type = ""
                param_type += "weight" if idx%2==0 else "bias"
                torch_param_name = layer + "." + param_type
                torch_param_dict[torch_param_name] = torch.from_numpy(param.data[...])

        print(dict(self.model.named_parameters()).keys())
        for pname, param in self.model.named_parameters():
            param.data = torch_param_dict[pname]
            print(f"Get parameter of {pname} from caffe, with shape of {param.shape}")
        if save:
            save_dir = "/home/acb11711tx/lzq/VideoPerturb2/models/ucf101_24_vgg16lstm.pt"
            self.save_weights(save_dir)
            print(f"Successfully save pamameters to {save_dir}.")

    def transfer_to_caffe (self, caffe_prototxt, caffemodel_save_dir=None):
        print("start")
        import caffe
        net = caffe.Net(caffe_prototxt, caffe.TEST)

        torch_param_dic = dict(self.model.named_parameters())
        if "module" in list(torch_param_dic.keys())[0]:
            torch_param_dic = {name[7:]: torch_param_dic[name] for name in torch_param_dic.keys()}
        print(torch_param_dic.keys())

        for layer, params in net.params.items():
            print(layer)
            layer = layer.replace("-", "_")
            for idx, param in enumerate(params):
                if "lstm" in layer:
                    param_type = "fc_x." if idx<=1 else "fc_h."
                else:
                    param_type = ""
                param_type += "weight" if idx%2==0 else "bias"
                torch_param_name = layer + "." + param_type
                param.data[...] = torch_param_dic[torch_param_name].detach().cpu().numpy()
                print(f"Get parameter of {torch_param_name} from pytorch, with shape of {param.shape}")
        if caffemodel_save_dir is not None:
            net.save(caffemodel_save_dir)

    def train_model(self, dataloaders, criterion, optimizer, checkpoint_name, num_epochs):
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
                        loss = criterion(torch.log(outputs), labels)    # Using NLLLoss

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
        # for samples in tqdm(dataloader):
        video_pred_dict = {}
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
            probs = process_activations(outputs, labels, softmaxed=True)[0]
            for bidx in range(bs):
                video_name = samples[2][bidx].split("/")[-1]
                label = labels.data[bidx]
                video_pred_dict[video_name] = probs.detach().cpu()[bidx].item()
                # print(f"{video_name}: {video_pred_dict[video_name]}")

        video_pred_dict = {name: pred for name, pred in sorted(video_pred_dict.items(), key=lambda item: item[1], reverse=True)}
        for video_name, pred in video_pred_dict.items():
            print(f"{video_name}: {pred:.3f}")

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
    from datasets.ucf101_24_dataset_vgg16lstm import UCF101_24_Dataset

    ds_path = "/home/acb11711tx/lzq/dataset/UCF101_24/"
    pt_save_dir = "/home/acb11711tx/lzq/VideoPerturb2/models/ucf101_24_vgg16lstm.pt"
    num_f = 8
    long_range = True

    multi_gpu = True
    num_devices = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ft = vgg16lstm(101)
    # model_ft.load_weights_from_caffemodel(save=False)
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