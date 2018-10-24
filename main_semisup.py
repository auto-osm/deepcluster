import argparse
import os
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from datetime import datetime

import models
import pickle

from utils.util import AverageMeter, UnifLabelSampler
from utils.ten_crop_and_finish import TenCropAndFinish
from models.new.sup_head5 import SupHead5
from utils.assess_acc_block import assess_acc_block
from utils.custom_cutout import custom_cutout
from PIL import Image

parser = argparse.ArgumentParser(
  description='PyTorch Implementation of DeepCluster')

parser.add_argument('--model_ind', type=int, required=True)
parser.add_argument('--old_model_ind', type=int, required=True)  # for features

# default is to use unlabelled (model 334)

parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/deepcluster")

parser.add_argument('--checkpoint_granularity', type=int, default=1)

# ----
parser.add_argument('--head_lr', default=0.05, type=float)

parser.add_argument('--trunk_lr', default=0.05, type=float)

parser.add_argument("--random_affine", default=False, action="store_true")
parser.add_argument("--affine_p", type=float, default=0.5)

parser.add_argument("--cutout", default=False, action="store_true")
parser.add_argument("--cutout_p", type=float, default=0.5)
parser.add_argument("--cutout_max_box", type=float, default=0.5)

parser.add_argument('--total_epochs', type=int, default=3200,
                    help='number of total epochs to run (default: 200)')

parser.add_argument('--seed', type=int, default=31,
                    help='random seed (default: 31)')
parser.add_argument('--verbose', action='store_true', help='chatty')

# means, std
_DATASET_NORM = {
  "STL10": (
  [0.45532353, 0.43217013, 0.3928851], [0.25528341, 0.24733134, 0.25604967]),
  "CIFAR10": (
  [0.49186879, 0.48265392, 0.44717729], [0.24697122, 0.24338894, 0.26159259]),
  "CIFAR20": (
  [0.50736205, 0.48668957, 0.44108858], [0.26748816, 0.2565931, 0.27630851]),
  "MNIST": None
}


def main():
  global args
  args = parser.parse_args()

  args.out_dir = os.path.join(args.out_root, str(args.model_ind))
  if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

  # get old args
  old_args_dir = os.path.join(args.out_root, str(args.old_model_ind))
  reloaded_args_path = os.path.join(old_args_dir, "config.pickle")
  print("Loading restarting args from: %s" % reloaded_args_path)
  with open(reloaded_args_path, "rb") as args_f:
    old_args = pickle.load(args_f)
  assert (args.old_model_ind == old_args.model_ind)
  next_epoch = 1

  if not hasattr(args, "if_stl_dont_use_unlabelled"):
    args.if_stl_dont_use_unlabelled = False

  sys.stdout.flush()

  # fix random seeds
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  np.random.seed(args.seed)

  args.epoch_acc = []
  args.epoch_loss = []

  # losses and acc
  fig, axarr = plt.subplots(2, sharex=False, figsize=(20, 20))

  # Data ---------------------------------------------------------------------

  # preprocessing of data
  tra = []
  tra_test = []
  if old_args.rand_crop_sz != -1:
    tra += [transforms.RandomCrop(old_args.rand_crop_sz)]
    tra_test += [transforms.CenterCrop(old_args.rand_crop_sz)]

  tra += [transforms.Resize(old_args.input_sz)]
  tra_test += [transforms.Resize(old_args.input_sz)]

  old_args.data_mean = None  # toggled on in cluster_assign
  old_args.data_std = None
  if old_args.normalize:
    data_mean, data_std = _DATASET_NORM[old_args.dataset]
    old_args.data_mean = data_mean
    old_args.data_std = data_std
    normalize = transforms.Normalize(mean=old_args.data_mean,
                                     std=old_args.data_std)
    tra.append(normalize)
    tra_test.append(normalize)

  # actual augmentation here
  if not (old_args.dataset == "MNIST"):
    tra += [transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.125)
            ]

  else:
    print("skipping horizontal flipping and jitter")

  if args.random_affine:
    print("adding affine with p %f" % args.affine_p)
    tra_test.append(transforms.RandomApply(
      [transforms.RandomAffine(18,
                               scale=(0.9, 1.1),
                               translate=(0.1, 0.1),
                               shear=10,
                               resample=Image.BILINEAR,
                               fillcolor=0)], p=args.affine_p)
    )

  if args.cutout:
    print("adding cutout with p %f max box %f" % (args.cutout_p,
                                                  args.cutout_max_box))
    # https://github.com/uoguelph-mlrg/Cutout/blob/master/images/cutout_on_cifar10.jpg
    tra_test.append(
        transforms.RandomApply(
            [custom_cutout(min_box=int(old_args.input_sz * 0.2),
                           max_box=int(old_args.input_sz *
                                      args.cutout_max_box))],
            p=args.cutout_p)
    )

  tra += [transforms.ToTensor()]
  #tra_test += [transforms.ToTensor()] # done in TenCropAndFinish

  tra = transforms.Compose(tra)
  tra_test = transforms.Compose(tra_test)

  assert (old_args.dataset == "STL10")
  dataset_class = datasets.STL10
  train_data = dataset_class(
    root=old_args.dataset_root,
    transform=tra,
    split="train")

  train_loader = torch.utils.data.DataLoader(train_data,
                                             batch_size=old_args.batch_sz,
                                             shuffle=True,
                                             num_workers=0,
                                             drop_last=False)

  test_data = dataset_class(
    root=old_args.dataset_root,
    transform=tra_test,
    split="test")
  test_data = TenCropAndFinish(test_data, input_sz=old_args.input_sz)
  contiguous_sz = 10

  test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=old_args.batch_sz,
                                            shuffle=False,
                                            num_workers=0,
                                            drop_last=False)

  # Model --------------------------------------------------------------------

  # features
  if args.verbose:
    print('Architecture: {}'.format(old_args.arch))
    sys.stdout.flush()
  features = models.__dict__[old_args.arch](sobel=old_args.sobel,
                                            out=old_args.k,
                                            input_sp_sz=old_args.input_sz,
                                            input_ch=old_args.input_ch)
  assert (features.top_layer is None)
  # remove top_layer parameters from checkpoint
  checkpoint = torch.load(os.path.join(old_args.out_dir, "%s.pytorch" %
                                       "best"))
  for key in checkpoint['state_dict']:
    if 'top_layer' in key:
      del checkpoint['state_dict'][key]
  features.load_state_dict(checkpoint['state_dict'])

  # wrap features in suphead
  print("old gt_k is: %d" % old_args.gt_k)
  model = SupHead5(features, dlen=features.dlen, gt_k=old_args.gt_k)

  # model = torch.nn.DataParallel(model)
  model.cuda()
  cudnn.benchmark = True

  # create optimizers
  opt_trunk = torch.optim.Adam(
    model.trunk.parameters(),
    lr=args.trunk_lr
  )
  opt_head = torch.optim.Adam(
    model.head.parameters(),
    lr=(args.head_lr)
  )

  # define loss function
  criterion = nn.CrossEntropyLoss().cuda()

  print("Doing pre assessment")
  sys.stdout.flush()
  acc = assess_acc_block(model, test_loader, gt_k=old_args.gt_k,
                         contiguous_sz=contiguous_sz)
  print("got %f" % acc)
  sys.stdout.flush()

  args.epoch_acc.append(acc)

  # Train --------------------------------------------------------------------
  for epoch in range(next_epoch, args.total_epochs):
    # train network with clusters as pseudo-labels
    loss = train(train_loader, model, criterion, opt_trunk, opt_head, epoch,
                 per_batch=(epoch == next_epoch))

    # assess ---------------------------------------------------------------

    acc = assess_acc_block(model, test_loader, gt_k=old_args.gt_k,
                           contiguous_sz=contiguous_sz)

    print("Model %d, epoch %d, train loss %f, acc %f, time %s"
          % (args.model_ind, epoch, loss, acc, datetime.now()))
    sys.stdout.flush()

    # update args
    is_best = False
    if acc > max(args.epoch_acc):
      is_best = True

    args.epoch_acc.append(acc)
    args.epoch_loss.append(loss)

    # draw graphs and save
    axarr[0].clear()
    axarr[0].plot(args.epoch_acc)
    axarr[0].set_title("Acc")

    axarr[1].clear()
    axarr[1].plot(args.epoch_loss)
    axarr[1].set_title("Training loss")

    # save -----------------------------------------------------------------
    # graphs
    fig.canvas.draw_idle()
    fig.savefig(os.path.join(args.out_dir, "plots.png"))

    # model
    if epoch % args.checkpoint_granularity == 0:
      torch.save({'state_dict': model.state_dict(),
                  'opt_trunk': opt_trunk.state_dict(),
                  'opt_head': opt_head.state_dict()},
                 os.path.join(args.out_dir, "latest.pytorch"))

      args.epoch = epoch  # last saved checkpoint

    if is_best:
      torch.save({'state_dict': model.state_dict(),
                  'opt_trunk': opt_trunk.state_dict(),
                  'opt_head': opt_head.state_dict()},
                 os.path.join(args.out_dir, "best.pytorch"))

      args.best_epoch = epoch

    # args
    with open(os.path.join(args.out_dir, "config.pickle"), 'w') as outfile:
      pickle.dump(args, outfile)

    with open(os.path.join(args.out_dir, "config.txt"), "w") as text_file:
      text_file.write("%s" % args)


def train(loader, model, criterion, opt_trunk, opt_head, epoch,
          per_batch):
  losses = AverageMeter()

  # switch to train mode
  model.train()

  if per_batch:
    print("num batches: %d" % len(loader))

  for i, (input_tensor, target) in enumerate(loader):
    opt_trunk.zero_grad()
    opt_head.zero_grad()

    input_var = torch.autograd.Variable(input_tensor.cuda())
    target_var = torch.autograd.Variable(target.cuda())

    output = model(input_var)

    loss = criterion(output, target_var)

    # compute gradient and do gradient step
    loss.backward()
    opt_trunk.step()
    opt_head.step()

    # record loss
    losses.update(float(loss.data), input_tensor.size(0))

    if ((i % 100) == 0) or per_batch:
      print("... epoch %d batch %d train loss %f time %s" %
            (epoch, i, float(loss.data), datetime.now()))
      sys.stdout.flush()

  return losses.avg

if __name__ == '__main__':
  main()
