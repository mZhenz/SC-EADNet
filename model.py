import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128, finetune=False):
        super(Model, self).__init__()
        self.finetune = finetune
        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, feature_dim, bias=True))

    def forward(self, x_0, x_1, x_2, x_3):
        x = torch.cat([x_0, x_1, x_2, x_3], dim=1)
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        if self.finetune:
            return feature
        else:
            out = self.g(feature)
            return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


if __name__ == '__main__':
    import time
    model = Model(feature_dim=128, finetune=True).cuda()
    model.eval()
    dummy_input = torch.tensor(torch.randn(2, 3, 31, 31)).cuda()
    t_start = time.time()
    for i in range(5124):
        with torch.no_grad():
            output = model(dummy_input, dummy_input, dummy_input, dummy_input)
    t_end = time.time()
    print(t_end-t_start)

