
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

import clustering
import models
from utils.util import AverageMeter, UnifLabelSampler, config_to_str, compute_acc
from utils.data import make_data, compute_data_stats
import pickle

from utils.ten_crop_and_finish import TenCropAndFinish

from sklearn.utils.linear_assignment_ import linear_assignment


parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

parser.add_argument('--model_ind', type=int, required=True)
parser.add_argument('--old_model_ind', type=int, required=True) # for features

parser.add_argument('--gt_k', type=int, required=True)

# default is to use unlabelled (model 334)

parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/deepcluster")

parser.add_argument('--checkpoint_granularity', type=int, default=1)

# ----
parser.add_argument('--head_lr', default=0.05, type=float)

parser.add_argument('--trunk_lr', default=0.05, type=float)

#parser.add_argument('--wd', default=-5, type=float,
#                    help='weight decay pow (default: -5)')

parser.add_argument('--workers', default=0, type=int,
                    help='number of data loading workers (default: 0)')
parser.add_argument('--total_epochs', type=int, default=3200,
                    help='number of total epochs to run (default: 200)')

#parser.add_argument('--momentum', default=0.9, type=float, help='momentum (
# default: 0.9)')

parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
parser.add_argument('--verbose', action='store_true', help='chatty')


# means, std
_DATASET_NORM = {
  "STL10": ([0.45532353, 0.43217013, 0.3928851], [0.25528341, 0.24733134, 0.25604967]),
  "CIFAR10": ([0.49186879, 0.48265392, 0.44717729], [0.24697122, 0.24338894, 0.26159259]),
  "CIFAR20": ([0.50736205, 0.48668957, 0.44108858], [0.26748816, 0.2565931, 0.27630851]),
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

    # losses and acc
    fig, axarr = plt.subplots(4, sharex=False, figsize=(20, 20))

    # distr
    distr_fig, distr_ax = plt.subplots(3, sharex=False, figsize=(20, 20))

    # Data ---------------------------------------------------------------------

    # preprocessing of data
    tra = []
    tra_test = []
    if old_args.rand_crop_sz != -1:
        tra += [transforms.RandomCrop(old_args.rand_crop_sz)]
        tra_test += [transforms.CenterCrop(old_args.rand_crop_sz)]

    tra += [transforms.Resize(old_args.input_sz)]
    tra_test += [transforms.Resize(old_args.input_sz)]

    old_args.data_mean = None # toggled on in cluster_assign
    old_args.data_std = None
    if old_args.normalize:
        data_mean, data_std = _DATASET_NORM[old_args.dataset]
        old_args.data_mean = data_mean
        old_args.data_std = data_std
        normalize = transforms.Normalize(mean=old_args.data_mean, std=old_args.data_std)
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

    tra += [transforms.ToTensor()]
    tra_test += [transforms.ToTensor()]

    tra = transforms.Compose(tra)
    tra_test = transforms.Compose(tra_test)

    assert(old_args.dataset == "STL10")
    dataset_class = datasets.STL10
    train_data = dataset_class(
        root=old_args.dataset_root,
        transform=tra,
        split="train")

    train_loader = torch.utils.data.DataLoader(train_data,
                                batch_size=old_args.batch_sz, # full batch
                                shuffle=True,
                                num_workers=0,
                                drop_last=False)

    test_data = dataset_class(
        root=old_args.dataset_root,
        transform=tra_test,
        split="test")
    test_data = TenCropAndFinish(test_data, input_sz=old_args.input_sz)

    test_loader = torch.utils.data.DataLoader(test_data,
                                batch_size=old_args.batch_sz, # full batch
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
                                       input_sp_sz=old_args.input_sz, input_ch=old_args.input_ch)
    assert(features.top_layer is None)
    # remove top_layer parameters from checkpoint
    checkpoint = torch.load(os.path.join(old_args.out_dir, "%s.pytorch" %
                                         "best"))
    for key in checkpoint['state_dict']:
        if 'top_layer' in key:
            del checkpoint['state_dict'][key]
    features.load_state_dict(checkpoint['state_dict'])

    # wrap features in suphead
    model = None

    #model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    # create optimizers

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    print("Doing some assessment")
    sys.stdout.flush()
    acc, distribution, centroid_min_max, assess_cluster_loss = \
        assess_acc(test_dataset, test_dataloader, model,
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
        #model.remove_feature_head_relu()

        # get the features for the whole dataset
        features = compute_features(dataloader, model, len(dataset))

        # cluster the features
        clustering_loss = deepcluster.cluster(features,
                                              proc_feat=args.proc_feat,
                                              verbose=args.verbose)

        # assign pseudo-labels to make new dataset
        train_dataset = clustering.cluster_assign(args,
                                                  deepcluster.images_lists,
                                                  dataset,
                                                  tra=tra)

        # uniformely sample per target
        sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)),
                                   deepcluster.images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_sz,
            num_workers=args.workers,
            sampler=sampler,
            pin_memory=True,
        )

        # prepare for training by reintroducing relu and resetting last layer
        #model.add_feature_head_relu()
        #model.reset_top_layer()

        # train network with clusters as pseudo-labels
        loss = train(train_dataloader, model, criterion, optimizer, epoch,
                     per_batch=(epoch == next_epoch))

        # assess ---------------------------------------------------------------

        acc, distribution, centroid_min_max, assess_cluster_loss = \
            assess_acc(test_dataset, test_dataloader, model, len(test_dataset))

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
                        'optimizer' : optimizer.state_dict()},
                       os.path.join(args.out_dir, "latest.pytorch"))

            args.epoch = epoch # last saved checkpoint

        if is_best:
            torch.save({'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()},
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

    # switch to train mode
    model.train()

    # only exists within this loop, not saved
    assert(not(model.top_layer is None))
    optimizer_tl = torch.optim.Adam(
        model.top_layer.parameters(),
        lr=args.lr,
        #weight_decay=10**args.wd,
    )

    if per_batch:
        print("num batches: %d" % len(loader))

    for i, (input_tensor, target) in enumerate(loader):
        opt.zero_grad()
        optimizer_tl.zero_grad()

        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target.cuda())

        output = model(input_var)

        loss = crit(output, target_var)

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
