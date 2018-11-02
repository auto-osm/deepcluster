# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
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
from datetime import datetime
import pickle
from sklearn.utils.linear_assignment_ import linear_assignment

import models
from utils.clustering.util import AverageMeter, UnifLabelSampler, \
  config_to_str
from utils.clustering.data import compute_data_stats

import clustering_segmentation
from utils.segmentation.loss import CustomCrossEntropyLoss
from utils.segmentation.data import make_data_segmentation
from utils.segmentation.util import compute_spatial_features
from utils.segmentation.assess_acc import assess_acc_segmentation

parser = argparse.ArgumentParser(
  description='PyTorch Implementation of DeepCluster')

parser.add_argument('--model_ind', type=int, required=True)
parser.add_argument('--k', type=int, required=True)
parser.add_argument('--gt_k', type=int, required=True)

parser.add_argument('--rand_crop_sz', type=int, default=-1)
parser.add_argument('--input_sz', type=int, required=True)

parser.add_argument('--normalize', action='store_true', default=False)

parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--dataset_root", type=str, required=True)

parser.add_argument('--batch_sz', default=256, type=int,
                    help='mini-batch size (default: 256)')

parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/deepcluster")

parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--resume_mode', type=str, choices=['latest', 'best'],
                    default="latest")
parser.add_argument('--checkpoint_granularity', type=int, default=1)

parser.add_argument('--find_data_stats', action='store_true', default=False)
parser.add_argument('--just_analyse', action='store_true', default=False)

parser.add_argument('--proc_feat', action='store_true', default=False)

# ----

parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['deepcluster_net10a_seg'],
                    default='deepcluster_net10a_seg',
                    required=True)

parser.add_argument('--sobel_and_rgb', action='store_true',
                    help='Sobel and rgb')
parser.add_argument('--sobel', action='store_true',
                    help='Sobel only. If false and not sobel_and_rgb,'
                         'only use rgb(ir).')

parser.add_argument('--clustering', type=str, choices=['Kmeans'],
                    default='Kmeans')
parser.add_argument('--lr', default=0.05, type=float,
                    help='learning rate (default: 0.05)')

parser.add_argument('--reassign', type=float, default=1.,
                    help="""how many epochs of training between two consecutive
                    reassignments of clusters (default: 1)""")
parser.add_argument('--workers', default=0, type=int,
                    help='number of data loading workers (default: 0)')
parser.add_argument('--total_epochs', type=int, default=3200,
                    help='number of total epochs to run (default: 200)')

parser.add_argument('--seed', type=int, default=31,
                    help='random seed (default: 31)')
parser.add_argument('--verbose', action='store_true', help='chatty')

# TODO
# means, std
_DATASET_NORM = {
  "Potsdam": None,
  "Coco164kCuratedFew": None,
  "Coco164kCuratedFull": None
}

def main():
  global args
  args = parser.parse_args()

  args.out_dir = os.path.join(args.out_root, str(args.model_ind))
  if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

  if args.resume:
    # get old args
    old_args = args

    reloaded_args_path = os.path.join(old_args.out_dir, "config.pickle")
    print("Loading restarting args from: %s" % reloaded_args_path)
    with open(reloaded_args_path, "rb") as args_f:
      args = pickle.load(args_f)
    assert (args.model_ind == old_args.model_ind)
    args.resume = True

    next_epoch = args.epoch + 1  # indexed from 0, also = num epochs passed

    print("stored losses and accs lens %d %d %d, cutting to %d %d %d" %
          (len(args.epoch_loss),
           len(args.epoch_cluster_dist),
           len(args.epoch_acc),
           next_epoch,
           next_epoch,
           next_epoch + 1))

    args.epoch_loss = args.epoch_loss[:next_epoch]
    args.epoch_assess_cluster_loss = args.epoch_assess_cluster_loss[:next_epoch]

    args.epoch_cluster_dist = args.epoch_cluster_dist[:next_epoch]
    args.epoch_acc = args.epoch_acc[:(next_epoch + 1)]

    if not hasattr(args, "if_stl_dont_use_unlabelled"):
      args.if_stl_dont_use_unlabelled = False
  else:
    args.epoch_acc = []
    args.epoch_assess_cluster_loss = []

    args.epoch_cluster_dist = []
    args.epoch_loss = []  # train loss

    args.epoch_distribution = []
    args.epoch_centroid_min = []
    args.epoch_centroid_max = []

    next_epoch = 0

  if not args.find_data_stats:
    print("args/config:")
    print(config_to_str(args))

  sys.stdout.flush()

  # fix random seeds
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  np.random.seed(args.seed)

  # losses and acc
  fig, axarr = plt.subplots(4, sharex=False, figsize=(20, 20))

  # distr
  distr_fig, distr_ax = plt.subplots(3, sharex=False, figsize=(20, 20))

  # Data ---------------------------------------------------------------------

  if args.dataset == "Potsdam":
    assert(not args.sobel and (not args.sobel_and_rgb)) # IID experiment settings
    args.input_ch = 4 # rgbir
  elif "Coco" in args.dataset:
    # unlike image clustering script, extra sobel_and_rgb setting
    if args.sobel_and_rgb:
      args.input_ch = 5
    elif args.sobel: # just sobel
      args.input_ch = 2
    else: # just rgb
      args.input_ch = 3

  # preprocessing of data
  tra = []
  tra_test = []
  if args.rand_crop_sz != -1:
    tra += [transforms.RandomCrop(args.rand_crop_sz)]
    tra_test += [transforms.CenterCrop(args.rand_crop_sz)]

  tra += [transforms.Resize(args.input_sz)]
  tra_test += [transforms.Resize(args.input_sz)]

  args.data_mean = None
  args.data_std = None
  if args.normalize and (not args.find_data_stats):
    data_mean, data_std = _DATASET_NORM[args.dataset]
    args.data_mean = data_mean
    args.data_std = data_std
    normalize = transforms.Normalize(mean=args.data_mean, std=args.data_std)
    tra.append(normalize)
    tra_test.append(normalize)

  # actual augmentation here, consistent with IID experiment settings
  if "Coco" in args.dataset:
    tra += [transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.125)
            ]
  elif args.dataset == "Potsdam":
    tra += [transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                   saturation=0.1, hue=0.1)
            ]

  tra += [transforms.ToTensor()]
  tra_test += [transforms.ToTensor()]

  tra = transforms.Compose(tra)
  tra_test = transforms.Compose(tra_test)

  # load the data
  dataset, dataloader, test_dataset, test_dataloader = make_data_segmentation(
    args, tra, tra_test)

  if args.find_data_stats:
    print(args.dataset)
    print("train dataset mean, std: %s, %s" %
          compute_data_stats(dataloader, len(dataset)))
    print("test dataset mean, std: %s, %s" %
          compute_data_stats(test_dataloader, len(test_dataset)))
    exit(0)

  # Model --------------------------------------------------------------------

  # CNN
  if args.verbose:
    print('Architecture: {}'.format(args.arch))
    sys.stdout.flush()
  model = models.__dict__[args.arch](sobel_and_rgb=args.sobel_and_rgb,
                                     sobel=args.sobel,
                                     out=args.k,
                                     input_sp_sz=args.input_sz,
                                     input_ch=args.input_ch)
  fd = model.dlen
  # model.features = torch.nn.DataParallel(model.features)
  model.cuda()
  cudnn.benchmark = True

  # create optimizer
  # top layer not created at this point!
  assert (model.top_layer is None)
  optimizer = torch.optim.Adam(
    filter(lambda x: x.requires_grad, model.parameters()),
    lr=args.lr,
  )

  if args.resume:
    # remove top_layer parameters from checkpoint
    checkpoint = torch.load(os.path.join(old_args.out_dir, "%s.pytorch" %
                                         args.resume_mode))

    for key in checkpoint['state_dict']:
      if 'top_layer' in key:
        del checkpoint['state_dict'][key]

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

  # define loss function
  criterion = CustomCrossEntropyLoss

  # clustering algorithm to use
  deepcluster = clustering_segmentation.__dict__[args.clustering](args.k)

  if (not args.resume) or args.just_analyse:
    print("Doing some assessment")
    sys.stdout.flush()
    acc, distribution, centroid_min_max, assess_cluster_loss = \
      assess_acc_segmentation(args, test_dataset, test_dataloader, model,
                              len(test_dataset))
    print("got %f" % acc)
    sys.stdout.flush()

    if args.just_analyse:
      exit(0)

    args.epoch_acc.append(acc)
    args.epoch_assess_cluster_loss.append(assess_cluster_loss)
    args.epoch_distribution.append(list(distribution))
    args.epoch_centroid_min.append(centroid_min_max[0])
    args.epoch_centroid_max.append(centroid_min_max[1])

  # Train --------------------------------------------------------------------
  for epoch in range(next_epoch, args.total_epochs):
    # remove relu (getting features)
    # model.remove_feature_head_relu()

    # get the features for the whole training dataset (dataset)
    features, masks = compute_spatial_features(dataloader, model,
                                               len(dataset))

    # find gt_k dlen centroids (using vectorised, unmasked only)
    # and storing assessment for each pixel (retain shape/masks)
    clustering_loss = deepcluster.cluster(features, masks,
                                          proc_feat=args.proc_feat,
                                          verbose=args.verbose)

    # assign pseudo-labels to make new dataset
    # i.e. set flat labels for each image, n, h, w (applying stored
    # assessment to generate new dataset)
    # get masks from original dataset
    train_dataset = clustering_segmentation.cluster_assign(args,
                                              deepcluster.pseudolabelled_imgs,
                                              dataset,
                                              tra=tra)

    # randomly sample as an approximation of evenly distributed batches
    sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)),
                               deepcluster.images_lists)

    train_dataloader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.batch_sz,
      num_workers=args.workers,
      sampler=sampler,
      pin_memory=True,
    )

    if epoch == next_epoch:
      print("fd length: %d" % fd)

    # train network with clusters as pseudo-labels
    loss = train(train_dataloader, model, criterion, optimizer, epoch,
                 per_batch=(epoch == next_epoch))

    # assess ---------------------------------------------------------------

    acc, distribution, centroid_min_max, assess_cluster_loss = \
      assess_acc_segmentation(args, test_dataset, test_dataloader, model,
                              len(test_dataset))

    print("Model %d, epoch %d, cluster loss %f, train loss %f, acc %f "
          "time %s"
          % (args.model_ind, epoch, clustering_loss, loss, acc,
             datetime.now()))
    sys.stdout.flush()

    # update args
    is_best = False
    if acc > max(args.epoch_acc):
      is_best = True

    args.epoch_acc.append(acc)
    args.epoch_assess_cluster_loss.append(assess_cluster_loss)
    args.epoch_loss.append(loss)
    args.epoch_cluster_dist.append(clustering_loss)

    args.epoch_distribution.append(distribution)
    args.epoch_centroid_min.append(centroid_min_max[0])
    args.epoch_centroid_max.append(centroid_min_max[1])

    # draw graphs and save
    axarr[0].clear()
    axarr[0].plot(args.epoch_acc)
    axarr[0].set_title("Acc")

    axarr[1].clear()
    axarr[1].plot(args.epoch_loss)
    axarr[1].set_title("Training loss")

    axarr[2].clear()
    axarr[2].plot(args.epoch_cluster_dist)
    axarr[2].set_title("Cluster distance (train, k)")

    axarr[3].clear()
    axarr[3].plot(args.epoch_assess_cluster_loss)
    axarr[3].set_title("Cluster distance (assess, gt_k)")

    distr_ax[0].clear()
    epoch_distribution = np.array(args.epoch_distribution)
    for gt_c in xrange(args.gt_k):
      distr_ax[0].plot(epoch_distribution[:, gt_c])
    distr_ax[0].set_title("Prediction distribution")

    distr_ax[1].clear()
    distr_ax[1].plot(args.epoch_centroid_min)
    distr_ax[1].set_title("Centroid avg-of-abs: min")

    distr_ax[2].clear()
    distr_ax[2].plot(args.epoch_centroid_max)
    distr_ax[2].set_title("Centroid avg-of-abs: max")

    # save -----------------------------------------------------------------
    # graphs
    fig.canvas.draw_idle()
    fig.savefig(os.path.join(args.out_dir, "plots.png"))

    distr_fig.canvas.draw_idle()
    distr_fig.savefig(os.path.join(args.out_dir, "distribution.png"))

    # model
    if epoch % args.checkpoint_granularity == 0:
      torch.save({'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()},
                 os.path.join(args.out_dir, "latest.pytorch"))

      args.epoch = epoch  # last saved checkpoint

    if is_best:
      torch.save({'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()},
                 os.path.join(args.out_dir, "best.pytorch"))

      args.best_epoch = epoch

    # args
    with open(os.path.join(args.out_dir, "config.pickle"), 'w') as outfile:
      pickle.dump(args, outfile)

    with open(os.path.join(args.out_dir, "config.txt"), "w") as text_file:
      text_file.write("%s" % args)


def train(loader, model, crit, opt, epoch, per_batch=False):
  """Training of the CNN.
      Args:
          loader (torch.utils.data.DataLoader): Data loader
          model (nn.Module): CNN
          crit (torch.nn): loss
          opt: optimizer for every parameters with True
                                 requires_grad in model except top layer
          epoch (int)
  """
  losses = AverageMeter()

  model.set_new_top_layer()

  # switch to train mode
  model.train()

  # only exists within this loop, not saved
  assert (not (model.top_layer is None))
  optimizer_tl = torch.optim.Adam(
    model.top_layer.parameters(),
    lr=args.lr,
    # weight_decay=10**args.wd,
  )

  if per_batch:
    print("num batches: %d" % len(loader))

  for i, (input_tensor, mask, target) in enumerate(loader):
    opt.zero_grad()
    optimizer_tl.zero_grad()

    input_var = torch.autograd.Variable(input_tensor.cuda())
    target_var = torch.autograd.Variable(target.cuda())

    output = model(input_var)

    loss = crit(output, mask, target_var)

    # compute gradient and do gradient step
    loss.backward()
    opt.step()
    optimizer_tl.step()

    # record loss
    losses.update(float(loss.data), input_tensor.size(0))

    if ((i % 100) == 0) or per_batch:
      print("... epoch %d batch %d train loss %f time %s" %
            (epoch, i, float(loss.data), datetime.now()))
      sys.stdout.flush()

  return losses.avg

if __name__ == '__main__':
  main()
