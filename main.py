import argparse
import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.HSI_new import HSI
from EADNet import EADNet
from model import Model
from torchvision import transforms

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    # print(train_bar)
    for pos_1, pos_2, target in train_bar:
        pos0_0, pos0_1, pos0_2, pos0_3 = pos_1[0].cuda(non_blocking=True), pos_1[1].cuda(non_blocking=True), \
                                         pos_1[2].cuda(non_blocking=True), pos_1[3].cuda(non_blocking=True)
        pos1_0, pos1_1, pos1_2, pos1_3 = pos_2[0].cuda(non_blocking=True), pos_2[1].cuda(non_blocking=True), \
                                         pos_2[2].cuda(non_blocking=True), pos_2[3].cuda(non_blocking=True)
        # print(pos0_0.shape)
        feature_1, out_1 = net(pos0_0, pos0_1, pos0_2, pos0_3)
        feature_2, out_2 = net(pos1_0, pos1_1, pos1_2, pos1_3)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss [B]
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)

        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            data_0, data_1, data_2, data_3 = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True), \
                                             data[2].cuda(non_blocking=True), data[3].cuda(non_blocking=True)
            feature, out = net(data_0, data_1, data_2, data_3)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)

        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data_0, data_1, data_2, data_3 = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True), \
                                             data[2].cuda(non_blocking=True), data[3].cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            feature, out = net(data_0, data_1, data_2, data_3)
            total_num += data[0].size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data[0].size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()
            # counts for each class
            one_hot_label = torch.zeros(data[0].size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            one_hot_label[:, 0] = 0
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data[0].size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)
            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


class MyCoTransform(object):
    def __init__(self, mode='train'):
        self.mode = mode
        pass

    def __call__(self, input0, input1, input2, input3):
        train_transform_0 = transforms.Compose([
            transforms.RandomResizedCrop(31),
            transforms.ToTensor(),
            # ip
            transforms.Normalize([0.23749262, 0.48883094, 0.4582705], [0.00555777, 0.00690145, 0.04093194])
            # sv
            # transforms.Normalize([0.34324876, 0.25490125, 0.40289667], [0.00757429, 0.0043157,  0.0362956])
            # houston 2013
            # transforms.Normalize([0.442528, 0.344754, 0.115389], [0.000489, 0.006495, 0.005109])
        ])

        train_transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(31),
            transforms.ToTensor(),
            # ip
            transforms.Normalize([0.23749262, 0.48883094, 0.4582705], [0.00555777, 0.00690145, 0.04093194])
            # sv
            # transforms.Normalize([0.34324876, 0.25490125, 0.40289667], [0.00757429, 0.0043157,  0.0362956])
            # houston 2013
            # transforms.Normalize([0.442528, 0.344754, 0.115389], [0.000489, 0.006495, 0.005109])
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            # indian pines
            transforms.Normalize([0.23749262, 0.48883094, 0.4582705], [0.00555777, 0.00690145, 0.04093194])
            # salinas
            # transforms.Normalize([0.34324876, 0.25490125, 0.40289667], [0.00757429, 0.0043157, 0.0362956])
            # houston 2013
            # transforms.Normalize([0.442528, 0.344754, 0.115389], [0.000489, 0.006495, 0.005109])
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


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs

    train_data = HSI(root_dir='./dataset/HSIdata/ip_50p_17c/', mode='train',
                     transform=[MyCoTransform(mode='train_0'), MyCoTransform(mode='train_1')])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)

    memory_data = HSI(root_dir='./dataset/HSIdata/ip_50p_17c/', mode='train',
                      transform=[MyCoTransform(mode='test'), MyCoTransform(mode='test')])
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    test_data = HSI(root_dir='./dataset/HSIdata/ip_50p_17c/', mode='val',
                    transform=[MyCoTransform(mode='test'), MyCoTransform(mode='test')])
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    # model setup and optimizer config
    
    # model = Model(feature_dim=feature_dim, finetune=False).cuda()
    model = EADNet(cl_dim=feature_dim, finetune=False).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = 17

    # training loop
    exp_id = 'sceadnet'
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}_{}'.format('ip', feature_dim, temperature, batch_size, epochs, exp_id)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('result/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'result/{}_model.pth'.format(save_name_pre))
