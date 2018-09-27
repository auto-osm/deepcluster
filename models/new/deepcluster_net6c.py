import torch.nn as nn
from vgg import VGGTrunk, VGGNet

# 4h but for cifar, 24x24

class ClusterNet6cTrunk(VGGTrunk):
  def __init__(self, config):
    super(ClusterNet6cTrunk, self).__init__()

    self.conv_size = 5
    self.pad = 2
    self.cfg = ClusterNet6c.cfg
    self.in_channels = config.in_channels if hasattr(config, 'in_channels') \
      else 3

    self.features = self._make_layers()

  def forward(self, x):
    x = self.features(x)
    bn, nf, h, w = x.size()
    x = x.view(bn, nf * h * w)
    return x

class ClusterNet6cHead(nn.Module):
  def __init__(self, config):
    super(ClusterNet6cHead, self).__init__()
    self.num_heads = config.num_heads

    self.cfg = ClusterNet6c.cfg
    num_features = self.cfg[-1][0]
    self.heads = nn.ModuleList([nn.Sequential(
      nn.Linear(num_features * 3 * 3, config.output_k),
      nn.Softmax(dim=1)) for _ in xrange(self.num_heads)])

  def forward(self, x, kmeans_use_features=False):
    results = []
    for i in xrange(self.num_heads):
      if kmeans_use_features:
        results.append(x)  # duplicates
      else:
        results.append(self.heads[i](x))
    return results

class ClusterNet6c(VGGNet):
  cfg = [(64, 1), ('M', None), (128, 1), ('M', None),
           (256, 1), ('M', None), (512, 1)]

  def __init__(self, config):
    super(ClusterNet6c, self).__init__()

    self.trunk = ClusterNet6cTrunk(config)
    self.head = ClusterNet6cHead(config)

    self._initialize_weights()

  def forward(self, x, kmeans_use_features=False):
    x = self.trunk(x)
    x = self.head(x, kmeans_use_features=kmeans_use_features) # returns list
    return x