import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class ConcatConv(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-05)
        self.prelu = nn.PReLU(noutput)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], dim=1)
        output = self.bn(output)
        return self.prelu(output)


class EADC(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        cha_group = int(chann / 8)

        self.conv1x1_1 = nn.Conv2d(chann, cha_group, 1, stride=1, padding=0, bias=True)
        self.conv1x1_2 = nn.Conv2d(chann, cha_group, 1, stride=1, padding=0, bias=True)
        self.conv1x1_3 = nn.Conv2d(chann, cha_group, 1, stride=1, padding=0, bias=True)
        self.conv1x1_4 = nn.Conv2d(chann, cha_group, 1, stride=1, padding=0, bias=True)

        self.conv3x1_1 = nn.Conv2d(cha_group, cha_group, (3, 1), stride=1, padding=(1, 0), bias=True )
        self.conv1x3_1 = nn.Conv2d(cha_group, cha_group, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.conv3x1_2 = nn.Conv2d(cha_group, cha_group, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1), groups=cha_group)
        self.conv1x3_2 = nn.Conv2d(cha_group, cha_group, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated), groups=cha_group)

        self.conv3x1_3 = nn.Conv2d(cha_group, cha_group, (3, 1), stride=1, padding=(2 * dilated, 0), bias=True,
                                   dilation=(dilated*2, 1), groups=cha_group)
        self.conv1x3_3 = nn.Conv2d(cha_group, cha_group, (1, 3), stride=1, padding=(0, 4 * dilated), bias=True,
                                   dilation=(1, dilated * 4), groups=cha_group)

        self.conv3x1_4 = nn.Conv2d(cha_group, cha_group, (3, 1), stride=1, padding=(4 * dilated, 0), bias=True,
                                   dilation=(dilated*4, 1), groups=cha_group)
        self.conv1x3_4 = nn.Conv2d(cha_group, cha_group, (1, 3), stride=1, padding=(0, 2 * dilated), bias=True,
                                   dilation=(1, dilated * 2), groups=cha_group)

        self.conv3x3 = nn.Conv2d(int(chann//2), int(chann//2), (3, 3), stride=1, padding=1, bias=True, groups=cha_group)
        self.bn3x3 = nn.BatchNorm2d(chann, eps=1e-05)

        self.conv1x1 = nn.Conv2d(chann, chann, 1, stride=1, padding=0)

        self.bn = nn.BatchNorm2d(chann, eps=1e-05)
        self.dropout = nn.Dropout2d(dropprob)
        self.prelu1 = nn.PReLU(cha_group)
        self.prelu2 = nn.PReLU(cha_group)
        self.prelu3 = nn.PReLU(cha_group)
        self.prelu4 = nn.PReLU(cha_group)

        self.prelu = nn.PReLU(chann)

    def forward(self, input):
        # MMRFC
        output = self.conv1x1_1(input)
        output = self.conv3x1_1(output)
        output = self.prelu1(output)
        output1 = self.conv1x3_1(output)

        output = self.conv1x1_2(input)
        output = self.conv3x1_2(output)
        output = self.prelu2(output)
        output2 = self.conv1x3_2(output)

        output = self.conv1x1_3(input)
        output = self.conv3x1_3(output)
        output = self.prelu3(output)
        output3 = self.conv1x3_3(output)

        output = self.conv1x1_4(input)
        output = self.conv3x1_4(output)
        output = self.prelu4(output)
        output4 = self.conv1x3_4(output)

        output = torch.cat([output1, output2, output3, output4], dim=1)
        output_transfer = self.conv3x3(output)

        x = torch.cat([output, output_transfer], dim=1)
        x = self.bn3x3(x)
        x = self.conv1x1(x)
        output = self.bn(x)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return self.prelu(output + input)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_block = ConcatConv(3, 16)
        # layer1: ConcatConv -> EADC*6
        self.layers1 = nn.ModuleList()
        self.layers1.append(ConcatConv(32, 64))
        for x in range(0, 6):  # 6 times
            self.layers1.append(EADC(64, 0.03, 1))
            # self.layers1.append(EADC(64, 0.1, 1))

        # layer2: ConcatConv -> EADC*9
        self.layers2 = nn.ModuleList()
        self.layers2.append(ConcatConv(64, 128))

        for x in range(0, 3):  # 3 times
            self.layers2.append(EADC(128, 0.03, 2))
            self.layers2.append(EADC(128, 0.03, 4))
            self.layers2.append(EADC(128, 0.03, 6))

    def forward(self, input):
        # output = self.initial_block(input)
        output2x = input
        output = input
        for layer in self.layers1:
            output = layer(output)
        output4x = output

        for layer in self.layers2:
            output = layer(output)
        output8x = output

        return output8x, output4x, output2x


class EADNet(nn.Module):
    def __init__(self, cl_dim, finetune=False):  # use encoder to pass pretrained encoder
        super().__init__()
        self.num_classes = 17
        self.finetune = finetune
        self.concatconv1 = ConcatConv(3, 8)
        self.concatconv2 = ConcatConv(3, 8)
        self.concatconv3 = ConcatConv(3, 8)
        self.concatconv4 = ConcatConv(3, 8)

        self.encoder = Encoder()
        self.downplooling1_1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, groups=16)
        self.downplooling1_2 = nn.Conv2d(16, 16, 3, 2, 1, groups=16)
        self.downplooling2 = nn.Conv2d(64, 64, 3, 2, 1, groups=64)
        self.output_conv = nn.Conv2d(in_channels=128+64+16, out_channels=128,
                                     kernel_size=1, stride=1,
                                     padding=0, bias=True)

        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, cl_dim, bias=True))

    def forward(self, input1, input2, input3, input4):
        input_1 = self.concatconv1(input1)
        input_2 = self.concatconv2(input2)
        input_3 = self.concatconv3(input3)
        input_4 = self.concatconv4(input4)
        input = torch.cat([input_1, input_2, input_3, input_4], dim=1)

        output8x, output4x, output2x = self.encoder(input)  # predict=False by default
        output1 = output8x
        output2 = self.downplooling2(output4x)
        output3 = self.downplooling1_1(output2x)
        output3 = self.downplooling1_2(output3)
        output = torch.cat([output1, output2, output3], dim=1)
        output = self.output_conv(output)
        feature = torch.flatten(output, start_dim=1)
        if self.finetune:
            return feature
        else:
            out = self.g(feature)
            return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

