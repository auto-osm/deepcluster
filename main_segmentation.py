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
from utils.segmentation.transforms import sobel_process

import clustering_segmentation
from utils.segmentation.data import make_data_segmentation
from utils.segmentation.util import compute_vectorised_features
from utils.segmentation.assess_acc import assess_acc_segmentation

parser = argparse.ArgumentParser(
  description='PyTorch Implementation of DeepCluster')

parser.add_argument('--model_ind', type=int, required=True)
parser.add_argument('--k', type=int, required=True)
parser.add_argument('--gt_k', type=int, required=True)

parser.add_argument('--input_sz', type=int, required=True)

parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--dataset_root", type=str, required=True)
parser.add_argument('--use_coarse_labels', action='store_true', default=False)

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

parser.add_argument('--max_num_pixel_samples', type=int, default=100000000)

parser.add_argument('--debug_by_using_test', action='store_true', default=False)

parser.add_argument("--no_pre_eval", default=False, action="store_true")

# Coco options
parser.add_argument('--include_things_labels', action='store_true', default=False)
parser.add_argument('--incl_animal_things', action='store_true', default=False)
parser.add_argument("--fine_to_coarse_dict", type=str,
                    default="/users/xuji/iid/iid_private/code/datasets"
                            "/segmentation/util/out/fine_to_coarse_dict.pickle")
parser.add_argument("--coco_164k_curated_version", type=int, default=-1)

# ----

parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['DeepclusterSegmentationNet10a'],
                    default='DeepclusterSegmentationNet10a',
                    required=True)

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

# -------------

parser.add_argument('--do_sobel', action='store_true', default=False)
parser.add_argument('--do_rgb', action='store_true', default=False)

parser.add_argument("--pre_scale_all", default=False, action="store_true") # new
parser.add_argument("--pre_scale_factor", type=float, default=0.5) #

parser.add_argument("--jitter_brightness", type=float, default=0.4)
parser.add_argument("--jitter_contrast", type=float, default=0.4)
parser.add_argument("--jitter_saturation", type=float, default=0.4)
parser.add_argument("--jitter_hue", type=float, default=0.125)

# flip equivariance
parser.add_argument("--flip_p", type=float, default=0.5)

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
  assert(args.do_sobel or args.do_rgb)

  if args.dataset == "Potsdam":
    assert(not args.do_sobel and args.do_rgb) # IID experiment settings
    args.in_channels = 4 # rgbir
    args.using_IR = True
  elif "Coco" in args.dataset:
    # unlike image clustering script, extra sobel_and_rgb setting
    args.in_channels = 0
    if args.do_rgb: # new naming to avoid confusion with clustering script
      args.in_channels += 3
    if args.do_sobel:
      args.in_channels += 2
    args.using_IR = False

  if not args.debug_by_using_test:
    if "Coco" in args.dataset:
      args.train_partitions = ["train2017", "val2017"]
      args.test_partitions = ["train2017", "val2017"]
    elif args.dataset == "Potsdam":
      args.train_partitions = ["unlabelled_train", "labelled_train",
                                 "labelled_test"]
      args.test_partitions = ["labelled_train", "labelled_test"]
    else:
      assert (False)
  else:
    if "Coco" in args.dataset:
      args.train_partitions = ["val2017"]
      args.test_partitions = ["val2017"]
    elif args.dataset == "Potsdam":
      args.train_partitions = ["labelled_test"]
      args.test_partitions = ["labelled_test"]
    else:
      assert (False)

  # load the data
  # transforms consistent with other experiments are taken care of within the
  # dataset, which gets passed the settings
  dataset, dataloader, test_dataset, test_dataloader = make_data_segmentation(
    args)

  # Model --------------------------------------------------------------------

  # CNN
  if args.verbose:
    print('Architecture: {}'.format(args.arch))
    sys.stdout.flush()
  model = models.__dict__[args.arch](args)
  fd = model.dlen

  if args.resume:
    # remove top_layer parameters from checkpoint
    checkpoint = torch.load(os.path.join(old_args.out_dir, "%s.pytorch" %
                                         args.resume_mode))

    for key in checkpoint['state_dict']:
      if 'top_layer' in key:
        del checkpoint['state_dict'][key]

    model.load_state_dict(checkpoint['state_dict'])

  model.cuda()
  model = torch.nn.DataParallel(model)
  #cudnn.benchmark = True

  # create optimizer
  # top layer not created at this point!
  assert (model.module.top_layer is None)
  optimizer = torch.optim.Adam(
    filter(lambda x: x.requires_grad, model.module.parameters()),
    lr=args.lr,
  )

  if args.resume:
    optimizer.load_state_dict(checkpoint['optimizer'])

  # define loss function
  criterion = nn.CrossEntropyLoss().cuda()

  # clustering algorithm to use
  deepcluster = clustering_segmentation.__dict__[args.clustering](args.k)

  if (not args.no_pre_eval):
    if ((not args.resume) or args.just_analyse):
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
  else:
    # dummy
    print("using dummy pre-eval values")
    args.epoch_acc.append(-1)
    args.epoch_assess_cluster_loss.append(-1)
    args.epoch_distribution.append([-1 for _ in xrange(args.gt_k)])
    args.epoch_centroid_min.append(-1)
    args.epoch_centroid_max.append(-1)

  # Train --------------------------------------------------------------------
  for epoch in range(next_epoch, args.total_epochs):
    # remove relu (getting features)
    # model.remove_feature_head_relu()

    # get the features for the whole training dataset (dataset)
    features = compute_vectorised_features(args, dataloader, model,
                                           len(dataset))

    # find gt_k dlen centroids (using vectorised, unmasked only)
    # and storing assessment for each pixel (retain shape/masks)
    clustering_loss = deepcluster.cluster(args, features, dataloader,
                                          len(dataset), model,
                                          proc_feat=args.proc_feat,
                                          verbose=args.verbose)

    # assign pseudo-labels to make new dataset
    # i.e. set flat labels for each image, n, h, w (applying stored
    # assessment to generate new dataset)
    # get masks from original dataset
    train_dataset = clustering_segmentation.cluster_assign(
                                              deepcluster.pseudolabelled_x,
                                              dataset)

    # randomly sample as an approximation of evenly distributed batches
    train_dataloader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.batch_sz,
      num_workers=args.workers,
      shuffle=True
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
    if (epoch % args.checkpoint_granularity == 0) or is_best:
      model.module.cpu()
      if epoch % args.checkpoint_granularity == 0:
        torch.save({'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   os.path.join(args.out_dir, "latest.pytorch"))

        args.epoch = epoch  # last saved checkpoint

      if is_best:
        torch.save({'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   os.path.join(args.out_dir, "best.pytorch"))

        args.best_epoch = epoch

      model.module.cuda()

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

  model.module.set_new_top_layer()

  # switch to train mode
  model.module.train()

  # only exists within this loop, not saved
  assert (not (model.module.top_layer is None))
  optimizer_tl = torch.optim.Adam(
    model.module.top_layer.parameters(),
    lr=args.lr,
  )

  if per_batch:
    print("num batches: %d" % len(loader))

  for i, (imgs, masks, targets) in enumerate(loader):
    opt.zero_grad()
    optimizer_tl.zero_grad()

    assert(imgs.is_cuda and masks.is_cuda and targets.is_cuda)

    if args.do_sobel:
      imgs = sobel_process(imgs, args.do_rgb, using_IR=args.using_IR)

    x_out = model(imgs)

    assert(masks.dtype == torch.uint8)
    assert(targets.dtype == torch.int32)

    x_out = x_out.permute(0, 2, 3, 1)
    bn, h, w, dlen = x_out.shape
    x_out = x_out.view(bn * h * w, args.gt_k)
    targets = targets.view(bn * h * w)
    assert(targets.min() >= 0 and targets.max() < args.gt_k)

    loss_per_elem = crit(x_out, targets, reduction="none")
    assert(loss_per_elem.shape == (bn * h * w,))
    assert(masks.shape == loss_per_elem.shape)
    loss = loss_per_elem * masks # avoid masked_select for memory
    loss = loss.sum()

    # compute gradient and do gradient step
    loss.backward()
    opt.step()
    optimizer_tl.step()

    # record loss
    losses.update(float(loss.data), imgs.size(0))

    if ((i % 100) == 0) or per_batch:
      print("... epoch %d batch %d train loss %f time %s" %
            (epoch, i, float(loss.data), datetime.now()))
      sys.stdout.flush()

  return losses.avg

if __name__ == '__main__':
  main()
