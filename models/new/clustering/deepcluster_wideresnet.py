# From https://github.com/xternalz/WideResNet-pytorch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from residual import ResNet
from make_sobel import make_sobel

__all__ = [ 'deepcluster_wideresnet']

class WideResNetBasicBlock(nn.Module):
  def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
    super(WideResNetBasicBlock, self).__init__()
    self.bn1 = nn.BatchNorm2d(in_planes)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_planes)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                           padding=1, bias=False)
    self.droprate = dropRate
    self.equalInOut = (in_planes == out_planes)
    self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                           padding=0, bias=False) or None
  def forward(self, x):
    if not self.equalInOut:
      x = self.relu1(self.bn1(x))
    else:
      out = self.relu1(self.bn1(x))
    out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, training=self.training)
    out = self.conv2(out)
    return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
  def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
    super(NetworkBlock, self).__init__()
    self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
  def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
    layers = []
    for i in range(nb_layers):
      layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
    return nn.Sequential(*layers)
  def forward(self, x):
    return self.layer(x)

class WideResNetTrunk(nn.Module):
  def __init__(self, sobel, input_ch, input_sp_sz, depth=16, dropRate=0.0):
    super(WideResNetTrunk, self).__init__()
    widen_factor = 8
    nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
    assert((depth - 4) % 6 == 0)
    n = (depth - 4) / 6
    block = WideResNetBasicBlock
    # 1st conv before any network block
    self.conv1 = nn.Conv2d(input_ch, nChannels[0], kernel_size=3,
                           stride=1,
                           padding=1, bias=False)
    # 1st block
    self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
    # 2nd block
    self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
    # 3rd block
    self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
    # global average pooling and classifier
    self.bn1 = nn.BatchNorm2d(nChannels[3])
    self.relu = nn.ReLU(inplace=True)

    if input_sp_sz == 64:
      avg_pool_sz = 16
    elif input_sp_sz == 32:
      avg_pool_sz = 8
    print("avg_pool_sz %d" % avg_pool_sz)

    self.avgpool = nn.AvgPool2d(avg_pool_sz, stride=1) # to 1x1

    # self.fc = nn.Linear(nChannels[3], num_classes)
    self.nChannels = nChannels[3]

    self.use_sobel = sobel
    self.sobel = None

  def forward(self, x):
    if self.use_sobel:
      x = self.sobel(x)

    out = self.conv1(x)
    out = self.block1(out)
    out = self.block2(out)
    out = self.block3(out)
    out = self.relu(self.bn1(out))

    out = self.avgpool(out)

    out = out.view(out.size(0), -1)
    return out

class DeepClusterWideResNet(ResNet):
  def __init__(self, sobel=False, out=None, input_sp_sz=None, input_ch=None):
    # no saving of configs
    super(DeepClusterWideResNet, self).__init__()

    # features, used for pseudolabels
    self.features = WideResNetTrunk(sobel, input_ch, input_sp_sz)
    self.last_conv = self.features.nChannels
    self.dlen = 2048
    self.feature_head = nn.Sequential(
        nn.Linear(self.last_conv, self.dlen)
    )

    # used for training
    self.relu = nn.ReLU(True)
    self.dropout = nn.Dropout(0.5)
    self.out = out
    self.top_layer = None

    self._initialize_weights()

    if sobel:
      self.features.sobel = make_sobel()

  def forward(self, x, penultimate=False):
    x = self.features(x)
    x = self.feature_head(x)

    # used by features for pesudolabels and assessment
    if penultimate:
      return x

    # used for training
    x = self.dropout(self.relu(x))
    x = self.top_layer(x)
    return x

  def set_new_top_layer(self):
    # called each epoch, post-features
    self.top_layer = nn.Linear(self.dlen, self.out)
    self.top_layer.cuda()
    self.top_layer.weight.data.normal_(0, 0.01)
    self.top_layer.bias.data.zero_()

def deepcluster_wideresnet(sobel=False, out=None, input_sp_sz=None,
                         input_ch=None):
  return DeepClusterWideResNet(sobel, out, input_sp_sz, input_ch)