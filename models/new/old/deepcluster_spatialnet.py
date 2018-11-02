import torch.nn as nn

from models.new import VGGTrunk, VGGNet
from models.new import make_sobel

# for 24x24 or 64x64

__all__ = [ 'deepcluster_spatialnet']

class DeepClusterSpatialNetTrunk(VGGTrunk):
    def __init__(self, sobel, input_ch):
        super(DeepClusterSpatialNetTrunk, self).__init__()

        self.conv_size = 5
        self.pad = 2
        self.cfg = DeepClusterSpatialNet.cfg
        self.in_channels = input_ch

        self.use_sobel = sobel
        self.sobel = None

        # relu only used in training phase, not for kmeans
        self.features = self._make_layers(last_relu=False)

    def forward(self, x):
        if self.use_sobel:
            x = self.sobel(x)
        x = self.features(x)
        bn, nf, h, w = x.size()
        x = x.view(bn, nf * h * w)
        return x

class DeepClusterSpatialNet(VGGNet):

    # for mnist
    def __init__(self, sobel=False, out=None, input_sp_sz=None, input_ch=None):
        super(DeepClusterSpatialNet, self).__init__()

        assert(input_sp_sz == 32)
        DeepClusterSpatialNet.cfg = [(64, 1), ('M', None), (128, 1),
                                     ('M', None),  (256, 1), (512, 1),
                                     ('M', None)]

        if input_sp_sz == 32:
            self.feats_sp_sz = 4
        elif input_sp_sz == 24:
            self.feats_sp_sz = 3

        # used for kmeans, make the pseudolabels
        self.last_conv = 512
        self.dlen = self.last_conv * self.feats_sp_sz * self.feats_sp_sz
        self.features = DeepClusterSpatialNetTrunk(sobel, input_ch)

        # the top layer, giving outputs for training time
        self.relu = nn.ReLU()
        self.out = out
        self.top_layer = None

        self._initialize_weights()

        if sobel:
            self.features.sobel = make_sobel()

    def forward(self, x, penultimate=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        assert(x.size(1) == self.dlen)

        # used by assess code and features
        if penultimate:
            return x

        x = self.relu(x)
        x = self.top_layer(x)
        return x

    def make_top_layer(self):
        # callled once at start of script
        self.top_layer = nn.Linear(self.dlen, self.out)
        self.top_layer.cuda()

    def reset_top_layer(self):
        # called each epoch, post-features
        self.top_layer.weight.data.normal_(0, 0.01)
        self.top_layer.bias.data.zero_()

def deepcluster_spatialnet(sobel=False, out=None, input_sp_sz=None, input_ch=None):
    return DeepClusterSpatialNet(sobel, out, input_sp_sz, input_ch)