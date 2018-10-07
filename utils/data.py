import torchvision
import torch
from torch.utils.data import ConcatDataset
import numpy as np

def make_data(args, data_transform_train, data_transform_test):
  # return training dataset, training dataloader, and test dataloader

  target_transform = None

  if "STL10" == args.dataset:
    dataset_class = torchvision.datasets.STL10

    train_partitions = ["train", "test"] #["train+unlabeled", "test"]
    mapping_partitions = ["train", "test"] # labelled only

  elif "CIFAR" in args.dataset:
    if args.dataset == "CIFAR10":
      dataset_class = torchvision.datasets.CIFAR10

    elif args.dataset == "CIFAR100":
      dataset_class = torchvision.datasets.CIFAR100

    elif args.dataset == "CIFAR20":
      dataset_class = torchvision.datasets.CIFAR100
      target_transform = _cifar100_to_cifar20

    train_partitions = [True, False]
    mapping_partitions = [True, False]

  elif "MNIST" == args.dataset:
    dataset_class = torchvision.datasets.MNIST

    train_partitions = [True, False]
    mapping_partitions = [True, False]

  else:
    assert(False)

  train_dataset, train_dataloader = \
    _make_dataset_and_dataloader(args, dataset_class, train_partitions,
                                 data_transform=data_transform_train,
                                 target_transform=None) # targets not used

  mapping_dataset, mapping_dataloader = \
    _make_dataset_and_dataloader(args, dataset_class, mapping_partitions,
                                 data_transform=data_transform_test,
                                 target_transform=target_transform)

  return train_dataset, train_dataloader, mapping_dataset, mapping_dataloader

def _make_dataset_and_dataloader(args, dataset_class, partitions,
                                 data_transform, target_transform):
  # used to make both train and test (mapping) dataloaders

  imgs_list = []
  for partition in partitions:
    if "STL10" == args.dataset:
      imgs_curr = dataset_class(
        root=args.dataset_root,
        transform=data_transform,
        split=partition,
        target_transform=target_transform)
    else:
      imgs_curr = dataset_class(
        root=args.dataset_root,
        transform=data_transform,
        train=partition,
        target_transform=target_transform)
    imgs_list.append(imgs_curr)

  dataset = ConcatDataset(imgs_list)
  dataloader = torch.utils.data.DataLoader(dataset,
                                batch_size=args.batch_sz,
                                shuffle=False,
                                num_workers=args.workers,
                                drop_last=False)

  return dataset, dataloader

def _cifar100_to_cifar20(target):
  _dict = \
    {0: 4,
    1: 1,
    2: 14,
    3: 8,
    4: 0,
    5: 6,
    6: 7,
    7: 7,
    8: 18,
    9: 3,
    10: 3,
    11: 14,
    12: 9,
    13: 18,
    14: 7,
    15: 11,
    16: 3,
    17: 9,
    18: 7,
    19: 11,
    20: 6,
    21: 11,
    22: 5,
    23: 10,
    24: 7,
    25: 6,
    26: 13,
    27: 15,
    28: 3,
    29: 15,
    30: 0,
    31: 11,
    32: 1,
    33: 10,
    34: 12,
    35: 14,
    36: 16,
    37: 9,
    38: 11,
    39: 5,
    40: 5,
    41: 19,
    42: 8,
    43: 8,
    44: 15,
    45: 13,
    46: 14,
    47: 17,
    48: 18,
    49: 10,
    50: 16,
    51: 4,
    52: 17,
    53: 4,
    54: 2,
    55: 0,
    56: 17,
    57: 4,
    58: 18,
    59: 17,
    60: 10,
    61: 3,
    62: 2,
    63: 12,
    64: 12,
    65: 16,
    66: 12,
    67: 1,
    68: 9,
    69: 19,
    70: 2,
    71: 10,
    72: 0,
    73: 1,
    74: 16,
    75: 12,
    76: 9,
    77: 13,
    78: 15,
    79: 13,
    80: 16,
    81: 19,
    82: 2,
    83: 4,
    84: 6,
    85: 19,
    86: 5,
    87: 5,
    88: 8,
    89: 19,
    90: 18,
    91: 1,
    92: 2,
    93: 15,
    94: 6,
    95: 0,
    96: 17,
    97: 8,
    98: 14,
    99: 13}

  return _dict[target]


"""
STL10
batch shape: [256, 3, 64, 64]
train dataset mean, std: [0.45532353 0.43217013 0.3928851 ], [0.25528341 0.24733134 0.25604967]
batch shape: [256, 3, 64, 64]
test dataset mean, std: [0.45092908 0.43245607 0.40070766], [0.25320484 0.24809043 0.26007558]

CIFAR10
batch shape: [256, 3, 32, 32]
train dataset mean, std: [0.49186879 0.48265392 0.44717729], [0.24697122 0.24338894 0.26159259]
batch shape: [256, 3, 32, 32]
test dataset mean, std: [0.49186879 0.48265392 0.44717729], [0.24697122 0.24338894 0.26159259]

CIFAR20
batch shape: [256, 3, 32, 32]
train dataset mean, std: [0.50736205 0.48668957 0.44108858], [0.26748816 0.2565931  0.27630851]
batch shape: [256, 3, 32, 32]
test dataset mean, std: [0.50736205 0.48668957 0.44108858], [0.26748816 0.2565931  0.27630851]
"""
def compute_data_stats(dataloader, num_imgs):
    nb = len(dataloader)
    for i, (imgs_tensor, _) in enumerate(dataloader):
        if i == 0:
            print("batch shape: %s" % list(imgs_tensor.shape))
            bn, c, h, w = imgs_tensor.shape
            assert(c == 3 or c == 1)
            imgs = np.zeros((num_imgs, c, h, w))

        if i < nb - 1:
            imgs[i * bn: (i + 1) * bn, :, :, :] = imgs_tensor.numpy()
        else:
            imgs[i * bn:, :, :, :] = imgs_tensor.numpy()

    return np.mean(imgs, axis=(0, 2, 3)), np.std(imgs, axis=(0, 2, 3))
