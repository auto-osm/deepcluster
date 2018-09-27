import torch
import torch.nn as nn
import math
from random import random as rd

__all__ = [ 'VGG', 'vgg11']


class VGG(nn.Module):

    def __init__(self, features, num_classes, sobel, input_sp_sz=None):
        super(VGG, self).__init__()

        if input_sp_sz == 64 or input_sp_sz == 32:
            linear_sz = 2
        else:
            assert(False)

        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * linear_sz * linear_sz, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True)
        )
        self.top_layer = nn.Linear(4096, num_classes)
        self._initialize_weights()
        if sobel:
            grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0,0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1,0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

    def forward(self, x):
        print(x.shape)
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        print(x.shape)
        exit(0)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        for y,m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                #print(y)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(input_dim, batch_norm, input_sp_sz=None):
    layers = []
    in_channels = input_dim

    if input_sp_sz == 64:
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    elif input_sp_sz == 32:
        cfg = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    else:
        assert (False)

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg11(sobel=False, bn=True, out=None, input_sp_sz=None):
    dim = 2 + int(not sobel)
    model = VGG(make_layers(dim, bn, input_sp_sz), out, sobel, input_sp_sz)
    return model
