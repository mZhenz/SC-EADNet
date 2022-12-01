import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

# import utils
from model import Model
from EADNet import EADNet
from data.HSI_new import HSI
from torchvision import transforms


class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        self.f = EADNet(cl_dim=128, finetune=True)
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)
        # self.f.load_state_dict(torch.load(pretrained_path), strict=False)

    def forward(self, x0, x1, x2, x3):
        x = self.f(x0, x1, x2, x3)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data0, data1, target in data_bar:
            data0_0, data0_1, data0_2, data0_3 = data0[0].cuda(non_blocking=True), data0[1].cuda(non_blocking=True), \
                                                 data0[2].cuda(non_blocking=True), data0[3].cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            out = net(data0_0, data0_1, data0_2, data0_3)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data0[0].size(0)
            total_loss += loss.item() * data0[0].size(0)
            out[:, 0] = -100
            # print(out)
            prediction = torch.argsort(out, dim=-1, descending=True)
            # print(prediction[:, 0:1])
            # print(target.unsqueeze(dim=-1))
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


class MyCoTransform(object):
    def __init__(self, mode='train'):
        self.mode = mode

        pass

    def __call__(self, input0, input1, input2, input3):
        train_transform_0 = transforms.Compose([
            transforms.RandomResizedCrop(31),
            transforms.ToTensor(),
            ######## ip
            transforms.Normalize([0.23749262, 0.48883094, 0.4582705], [0.00555777, 0.00690145, 0.04093194])
            ######## sv
            # transforms.Normalize([0.34324876, 0.25490125, 0.40289667], [0.00757429, 0.0043157,  0.0362956])
        ])

        train_transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(31),
            transforms.ToTensor(),
            ########## ip
            transforms.Normalize([0.23749262, 0.48883094, 0.4582705], [0.00555777, 0.00690145, 0.04093194])
            ########## sv
            # transforms.Normalize([0.34324876, 0.25490125, 0.40289667], [0.00757429, 0.0043157,  0.0362956])
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            # indian pines
            transforms.Normalize([0.23749262, 0.48883094, 0.4582705], [0.00555777, 0.00690145, 0.04093194])
            # salinas
            # transforms.Normalize([0.34324876, 0.25490125, 0.40289667], [0.00757429, 0.0043157, 0.0362956])
        ])

        if self.mode == 'train_0':
            img0 = train_transform_0(input0)
            img1 = train_transform_0(input1)
            img2 = train_transform_0(input2)
            img3 = train_transform_0(input3)
        elif self.mode == 'train_1':
            img0 = train_transform_1(input0)
            img1 = train_transform_1(input1)
            img2 = train_transform_1(input2)
            img3 = train_transform_1(input3)
        elif self.mode == 'test':
            img0 = test_transform(input0)
            img1 = test_transform(input1)
            img2 = test_transform(input2)
            img3 = test_transform(input3)

        return [img0, img1, img2, img3]


def adjust_lr(optimizer, epoch, lr, w_decay, epochs):
    """
       POLY learning rate policy
    """
    # decay = 0.1**(sum(epoch >= np.array(lr_steps)))
    decay = ((1 - float(epoch) / epochs) ** (0.9))
    lr_ = lr * decay
    decay = w_decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
        param_group['weight_decay'] = decay


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='',
                        help='The pretrained model path')
    parser.add_argument('--datapath', type=str, default='./dataset/HSIdata/ip_5_17c/', help='Path of dataset')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--savepath', type=str, default='', help='save path')

    args = parser.parse_args()
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    data_path = args.datapath

    # from main import MyCoTransform
    train_data = HSI(root_dir=data_path, mode='train',
                     transform=[MyCoTransform(mode='train_0'), MyCoTransform(mode='train_0')])
    test_data = HSI(root_dir=data_path, mode='val',
                    transform=[MyCoTransform(mode='test'), MyCoTransform(mode='test')])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=200, shuffle=False, num_workers=16, pin_memory=True)

    model = Net(num_class=17, pretrained_path=None).cuda()
    for param in model.f.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_acc@1': []}

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        results['best_acc@1'].append(best_acc)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(args.savepath + '.csv', index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), args.savepath + '.pth')

    print('====>best accuracy:', best_acc)