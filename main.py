# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
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
from datetime import datetime

import clustering
import models
from utils.util import AverageMeter, UnifLabelSampler, config_to_str, compute_acc
from utils.data import make_data, compute_data_stats
import pickle

from sklearn.utils.linear_assignment_ import linear_assignment


parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

parser.add_argument('--model_ind', type=int, required=True)
parser.add_argument('--k', type=int, required=True)
parser.add_argument('--gt_k', type=int, required=True)

parser.add_argument('--resize_sz', type=int, default=None)
parser.add_argument('--crop_sz', type=int, required=True)

parser.add_argument('--dist_granularity', type=int, default=1000)

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
                    choices=['alexnet', 'vgg11', 'deepcluster_net6c',
                             'deepcluster_net5g'],
                    required=True)
parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                    default='Kmeans', help='clustering algorithm (default: Kmeans)')
parser.add_argument('--lr', default=0.05, type=float,
                    help='learning rate (default: 0.05)')
parser.add_argument('--wd', default=-5, type=float,
                    help='weight decay pow (default: -5)')
parser.add_argument('--reassign', type=float, default=1.,
                    help="""how many epochs of training between two consecutive
                    reassignments of clusters (default: 1)""")
parser.add_argument('--workers', default=0, type=int,
                    help='number of data loading workers (default: 0)')
parser.add_argument('--total_epochs', type=int, default=3200,
                    help='number of total epochs to run (default: 200)')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')

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

    if args.resume:
        # get old args
        old_args = args

        reloaded_args_path = os.path.join(old_args.out_dir, "args.pickle")
        print("Loading restarting args from: %s" % reloaded_args_path)
        with open(reloaded_args_path, "rb") as args_f:
          args = pickle.load(args_f)
        assert (args.model_ind == old_args.model_ind)
        args.resume = True

        next_epoch = args.epoch + 1 # indexed from 0, also = num epochs passed

        print("stored losses and accs lens %d %d %d, cutting to %d %d %d" %
              (len(args.epoch_loss),
               len(args.epoch_cluster_dist),
               len(args.epoch_acc),
               next_epoch,
               next_epoch,
               next_epoch + 1))

        args.epoch_loss = args.epoch_loss[:next_epoch]
        args.epoch_cluster_dist = args.epoch_cluster_dist[:next_epoch]
        args.epoch_acc = args.epoch_acc[:(next_epoch + 1)]

        if not hasattr(args, 'resize_sz'):
            args.resize_sz = None

        if not hasattr(args, 'resume_mode'):
            args.resume_mode = old_args.resume_mode

        if not hasattr(args, 'just_analyse'):
            args.just_analyse = old_args.just_analyse

        if not hasattr(args, 'proc_feat'):
            args.proc_feat = old_args.proc_feat
    else:
        args.epoch_acc = []
        args.epoch_cluster_dist = []
        args.epoch_loss = [] # train loss

        next_epoch = 0

    if not args.find_data_stats:
        print("args:")
        print(config_to_str(args))

    sys.stdout.flush()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # losses and acc
    fig, axarr = plt.subplots(3, sharex=False, figsize=(20, 20))

    # distr
    distr_fig, distr_ax = plt.subplots(1, figsize=(20, 20))

    # Data ---------------------------------------------------------------------

    if args.dataset == "MNIST":
        assert(not args.sobel)
        args.input_ch = 1
    else:
        if args.sobel:
            args.input_ch = 2
        else:
            args.input_ch = 3

    # preprocessing of data
    tra = []
    if args.resize_sz is not None:
        tra.append(transforms.Resize(args.resize_sz))
    tra += [transforms.RandomCrop(args.crop_sz),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]

    args.data_mean = None # toggled on in cluster_assign
    args.data_std = None
    if args.normalize and (not args.find_data_stats):
        data_mean, data_std = _DATASET_NORM[args.dataset]
        args.data_mean = data_mean
        args.data_std = data_std
        normalize = transforms.Normalize(mean=args.data_mean, std=args.data_std)
        tra.append(normalize)

    tra = transforms.Compose(tra)

    # load the data
    dataset, dataloader, test_dataset, test_dataloader = make_data(args, tra)

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
    model = models.__dict__[args.arch](sobel=args.sobel, out=args.k,
                                       input_sp_sz=args.crop_sz, input_ch=args.input_ch)
    fd = model.dlen
    #model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )

    if args.resume:
        # remove top_layer parameters from checkpoint
        checkpoint = torch.load(os.path.join(old_args.out_dir, "%s.pytorch" %
                                             args.resume_mode))
        """
        for key in checkpoint['state_dict']:
            if 'top_layer' in key:
                del checkpoint['state_dict'][key]
        """

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.k)

    if (not args.resume) or args.just_analyse:
        print("Doing some assessment")
        acc = assess_acc(test_dataset, test_dataloader, model,
                         len(test_dataset), fig_ax=(distr_fig, distr_ax),
                         ext="pre")
        print("got %f" % acc)
        sys.stdout.flush()

        if args.just_analyse:
            exit(0)

        args.epoch_acc.append(acc)

    # Train --------------------------------------------------------------------
    for epoch in range(next_epoch, args.total_epochs):
        # remove head
        #model.classifier = nn.Sequential(*list(model.classifier.children())[
        # :-1])

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
            #sampler=sampler,
            pin_memory=True,
        )

        # set last fully connected layer
        # top layer is created from new in each epoch! O_O
        """
        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        model.classifier = nn.Sequential(*mlp)
        """
        if epoch == next_epoch:
            print("fd length: %d" % fd)

        model.reset_top_layer()

        # train network with clusters as pseudo-labels
        loss = train(train_dataloader, model, criterion, optimizer, epoch,
                     per_batch=(epoch == next_epoch))

        # assess ---------------------------------------------------------------
        if epoch % args.dist_granularity == 0:
            dist_ext = str(epoch)
        else:
            dist_ext = ""

        acc = assess_acc(test_dataset, test_dataloader, model,
                         len(test_dataset), fig_ax=(distr_fig, distr_ax),
                         ext=dist_ext)

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
        args.epoch_loss.append(loss)
        args.epoch_cluster_dist.append(clustering_loss)

        # draw graphs and save
        axarr[0].clear()
        axarr[0].plot(args.epoch_acc)
        axarr[0].set_title("Acc")

        axarr[1].clear()
        axarr[1].plot(args.epoch_loss)
        axarr[1].set_title("Training loss")

        axarr[2].clear()
        axarr[2].plot(args.epoch_cluster_dist)
        axarr[2].set_title("Cluster distance")

        # save -----------------------------------------------------------------
        # graph
        fig.canvas.draw_idle()
        fig.savefig(os.path.join(args.out_dir, "plots.png"))

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
        with open(os.path.join(args.out_dir, "args.pickle"), 'w') as outfile:
            pickle.dump(args, outfile)

        with open(os.path.join(args.out_dir, "args.txt"), "w") as text_file:
            text_file.write("%s" % args)

def train(loader, model, crit, opt, epoch, per_batch=False):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    losses = AverageMeter()

    # switch to train mode
    model.train()

    """
    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=10**args.wd,
    )
    """

    if per_batch:
        print("num batches: %d" % len(loader))

    for i, (input_tensor, target) in enumerate(loader):

        # save checkpoint
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var)

        loss = crit(output, target_var)

        # record loss
        losses.update(float(loss.data), input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        #optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        #optimizer_tl.step()

        if ((i % 100) == 0) or per_batch:
            print("... epoch %d batch %d train loss %f time %s" %
                  (epoch, i, float(loss.data), datetime.now()))
            sys.stdout.flush()

    return losses.avg

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

def assess_acc(test_dataset, test_dataloader, model, num_imgs, fig_ax, ext=""):
    # new clusterer
    deepcluster = clustering.__dict__[args.clustering](args.gt_k)
    features = compute_features(test_dataloader, model, num_imgs)
    _ = deepcluster.cluster(features, proc_feat=args.proc_feat,
                            verbose=args.verbose)

    relabelled_test_dataset = clustering.cluster_assign(args,
                                             deepcluster.images_lists,
                                             test_dataset)

    assert(num_imgs == len(test_dataset))
    assert(num_imgs == len(relabelled_test_dataset))

    true_labels = np.array([test_dataset[i][1] for i in xrange(num_imgs)])
    predicted_labels = np.array([relabelled_test_dataset[i][1] for i in xrange(num_imgs)])
    # assuming the order corresponds to indices, for centroids
    analyse(predicted_labels, args.gt_k, fig_ax=fig_ax, ext=ext,
            names=get_sizes(deepcluster.centroids))

    assert(true_labels.min() == 0)
    assert(true_labels.max() == args.gt_k - 1)
    assert(predicted_labels.min() >= 0)
    assert(predicted_labels.max() < args.gt_k)

    # hungarian matching
    num_correct = np.zeros((args.gt_k, args.gt_k))
    for i in xrange(num_imgs):
      num_correct[predicted_labels[i], true_labels[i]] += 1
    match = linear_assignment(num_imgs - num_correct)

    reordered_preds = np.zeros(num_imgs)
    for pred_i, target_i in match:
        reordered_preds[predicted_labels == pred_i] = target_i

    #analyse(reordered_preds, args.gt_k, ext="reordered") shuld be same

    return compute_acc(reordered_preds, true_labels, args.gt_k)

def analyse(predictions, gt_k, fig_ax, ext="", names=None):
    # bar chart showing assignment per cluster centre (named)

    predictions = np.array(predictions)
    sums = np.array([sum(predictions == c) for c in xrange(gt_k)])

    sorted_indices = np.argsort(sums).astype("int")
    sums = list(sums[sorted_indices])

    assert(len(predictions) == sum(sums))
    fig, ax = fig_ax

    ax.clear()
    ax.bar(range(gt_k), sums, align='center', alpha=0.5)

    if names is not None:
        names = list(names[sorted_indices])
        assert(len(names) == len(sums))
        names = [str(c) for c in names]

        ax.set_xticks(range(gt_k))
        ax.set_xticklabels(names, rotation=45)

    ax.set_ylabel("Counts")
    ax.set_xlabel("Average abs value per centroid")

    ax.set_title("Cluster distribution (%s)" % ext)
    fig.canvas.draw_idle()
    fig.savefig(os.path.join(args.out_dir, "distribution_%s.png" % ext))

def get_sizes(centroids):
    # k, d matrix
    # e.g. 10, 3200 (stl10 with net5g)

    k, d = centroids.shape

    return centroids.sum(axis=1) / float(d)

if __name__ == '__main__':
    main()
