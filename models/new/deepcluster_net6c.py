import torch.nn as nn
from vgg import VGGTrunk, VGGNet
from make_sobel import make_sobel
# for 24x24

__all__ = [ 'deepcluster_net6c']

class DeepClusterNet6cTrunk(VGGTrunk):
    def __init__(self, sobel, input_ch):
        super(DeepClusterNet6cTrunk, self).__init__()

        self.conv_size = 5
        self.pad = 2
        self.cfg = DeepClusterNet6c.cfg
        self.in_channels = input_ch

        self.use_sobel = sobel
        self.sobel = None

        self.features = self._make_layers()

    def forward(self, x):
        if self.use_sobel:
            x = self.sobel(x)
        x = self.features(x)
        bn, nf, h, w = x.size()
        x = x.view(bn, nf * h * w)
        return x

class DeepClusterNet6c(VGGNet):
    cfg = [(64, 1), ('M', None), (128, 1), ('M', None),
           (256, 1), ('M', None), (512, 1)]

    def __init__(self, sobel=False, out=None, input_sp_sz=None, input_ch=None):
        super(DeepClusterNet6c, self).__init__()

        self.features = DeepClusterNet6cTrunk(sobel, input_ch)

        num_features = DeepClusterNet6c.cfg[-1][0]

        assert(input_sp_sz == 24)
        self.classifier = nn.Sequential(
            nn.Linear(num_features * 3 * 3, 4096),
            nn.ReLU(True)
        )

        self.top_layer = nn.Linear(4096, out)

        self._initialize_weights()

        if sobel:
            self.features.sobel = make_sobel()

    def forward(self, x, penultimate=False):
        x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        # used by assess code
        if penultimate:
            return x

        if self.top_layer:
            x = self.top_layer(x)
        return x

def deepcluster_net6c(sobel=False, out=None, input_sp_sz=None, input_ch=None):
    return DeepClusterNet6c(sobel, out, input_sp_sz, input_ch)