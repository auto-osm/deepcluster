import torch.nn as nn
from vgg import VGGTrunk, VGGNet
from make_sobel import make_sobel
# for 24x24 or 64x64

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

    def __init__(self, sobel=False, out=None, input_sp_sz=None, input_ch=None):
        super(DeepClusterNet6c, self).__init__()

        if input_sp_sz == 64:
            DeepClusterNet6c.cfg = [(64, 1), ('M', None), (128, 1), ('M', None),
                   (256, 1), ('M', None), (512, 1), ('M', None)]
            self.feats_sp_sz = 4
        elif input_sp_sz == 24:
            DeepClusterNet6c.cfg = [(64, 1), ('M', None), (128, 1), ('M', None),
                   (256, 1), ('M', None), (512, 1)]
            self.feats_sp_sz = 3

        self.features = DeepClusterNet6cTrunk(sobel, input_ch)

        """
        self.classifier = nn.Sequential(
            nn.Linear(num_features * 3 * 3, 4096),
            nn.ReLU(True)
        )
        """

        self.last_conv = 512
        self.dlen = 1000

        self.feature_head = nn.Sequential(
            nn.Linear(self.last_conv * self.feats_sp_sz * self.feats_sp_sz, self.dlen),
            nn.BatchNorm1d(self.dlen),
            nn.ReLU(True)
        )

        self.top_layer = nn.Linear(self.dlen, out)

        self._initialize_weights()

        if sobel:
            self.features.sobel = make_sobel()

    def forward(self, x, penultimate=False):
        x = self.features(x)
        x = self.feature_head(x)

        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)

        # used by assess code and features
        if penultimate:
            return x

        x = self.top_layer(x)
        return x

    def reset_top_layer(self):
        self.top_layer.weight.data.normal_(0, 0.01)
        self.top_layer.bias.data.zero_()
        self.top_layer.cuda()

def deepcluster_net6c(sobel=False, out=None, input_sp_sz=None, input_ch=None):
    return DeepClusterNet6c(sobel, out, input_sp_sz, input_ch)