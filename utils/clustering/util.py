# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

import models


def config_to_str(config):
  attrs = vars(config)
  string_val = "Config: -----\n"
  string_val += "\n".join("%s: %s" % item for item in attrs.items())
  string_val += "\n----------"
  return string_val


def load_model(path):
  """Loads model and return it without DataParallel table."""
  if os.path.isfile(path):
    print("=> loading checkpoint '{}'".format(path))
    checkpoint = torch.load(path)

    # size of the top layer
    N = checkpoint['state_dict']['top_layer.bias'].size()

    # build skeleton of the model
    sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
    model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

    # deal with a dataparallel table
    def rename_key(key):
      if not 'module' in key:
        return key
      return ''.join(key.split('.module'))

    checkpoint['state_dict'] = {rename_key(key): val
                                for key, val
                                in checkpoint['state_dict'].items()}

    # load weights
    model.load_state_dict(checkpoint['state_dict'])
    print("Loaded")
  else:
    model = None
    print("=> no checkpoint found at '{}'".format(path))
  return model


class UnifLabelSampler(Sampler):
  """Samples elements uniformely accross pseudolabels.
      Args:
          N (int): size of returned iterator.
          images_lists: dict of key (target), value (list of data with this 
          target)
  """

  def __init__(self, N, images_lists):
    self.N = N
    self.images_lists = images_lists
    self.indexes = self.generate_indexes_epoch()

  def generate_indexes_epoch(self):
    size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
    res = np.zeros(size_per_pseudolabel * len(self.images_lists))

    for i in range(len(self.images_lists)):
      indexes = np.random.choice(
        self.images_lists[i],
        size_per_pseudolabel,
        replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
      )
      res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

    np.random.shuffle(res)
    return res[:self.N].astype('int')

  def __iter__(self):
    return iter(self.indexes)

  def __len__(self):
    return self.N


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
  for param_group in optimizer.param_groups:
    lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
    param_group['lr'] = lr


class Logger():
  """ Class to update every epoch to keep trace of the results
  Methods:
      - log() log and save
  """

  def __init__(self, path):
    self.path = path
    self.data = []

  def log(self, train_point):
    self.data.append(train_point)
    with open(os.path.join(self.path), 'wb') as fp:
      pickle.dump(self.data, fp, -1)


def compute_features(dataloader, model, N):
  model.eval()
  # discard the label information in the dataloader
  for i, (input_tensor, _) in enumerate(dataloader):
    input_var = torch.autograd.Variable(input_tensor.cuda())
    with torch.no_grad():
      # penultimate = features
      aux = model(input_var, penultimate=True).data.cpu().numpy()

    if i == 0:
      features = np.zeros((N, aux.shape[1])).astype('float32')

    if i < len(dataloader) - 1:
      features[i * args.batch_sz: (i + 1) * args.batch_sz] = aux.astype(
        'float32')
    else:
      # special treatment for final batch
      features[i * args.batch_sz:] = aux.astype('float32')

  return features
