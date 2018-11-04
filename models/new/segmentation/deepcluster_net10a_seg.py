import torch.nn as nn
import math

import torch
import torch.nn.functional as F

from ..clustering.vgg import VGGTrunk, VGGNet

__all__ = ['DeepclusterSegmentationNet10a']

# From first iteration of code, based on VGG11:
# https://github.com/xu-ji/unsup/blob/master/mutual_information/networks/vggseg.py

class SegmentationNet10aTrunk(VGGTrunk):

  def __init__(self, config, cfg):
    super(SegmentationNet10aTrunk, self).__init__()

    assert(config.input_sz % 2 == 0)

    self.conv_size = 3
    self.pad = 1
    self.cfg = cfg
    self.in_channels = config.in_channels

    self.features = self._make_layers()

  def forward(self, x):
    x = self.features(x) # do not flatten
    return x

class DeepclusterSegmentationNet10a(VGGNet):
  cfg = [(64, 1), (128, 1), ('M', None), (256, 1), (256, 1),
         (512, 2), (512, 2)]  # 30x30 recep field

  def __init__(self, config):
    super(DeepclusterSegmentationNet10a, self).__init__()

    self.input_sz = config.input_sz
    self.features = SegmentationNet10aTrunk(config, cfg=DeepclusterSegmentationNet10a.cfg)

    self.last_conv = 512
    self.dlen = 1000

    # corrected, 0 padding head
    self.feature_head = nn.Sequential(nn.Conv2d(self.last_conv, self.dlen, kernel_size=1,
                stride=1, dilation=1, padding=0, bias=False),
                nn.BatchNorm2d(self.dlen))

    self.relu = nn.ReLU(True)
    self.out = config.k # not the same as gt_k
    self.top_layer = None

    self._initialize_weights()

  def forward(self, x, penultimate=False):
    x = self.features(x)
    x = self.feature_head(x)

    # used by assess code and features
    if not penultimate:
      x = self.top_layer(self.relu(x))

    x = F.interpolate(x, size=self.input_sz, mode="bilinear")
    return x

  def set_new_top_layer(self):
    # called each epoch, post-features
    self.top_layer = nn.Conv2d(self.dlen, self.out, kernel_size=1,
                stride=1, dilation=1, padding=0, bias=False)
    self.top_layer.cuda()
    nn.init.kaiming_normal_(self.top_layer.weight, mode="fan_in",
                            nonlinearity='relu') # consistent with other convs
    self.top_layer.bias.data.zero_()

