import sys
from datetime import datetime

import torch
from torch.utils.data import ConcatDataset

from datasets.segmentation import cocostuff
from datasets.segmentation import potsdam

def make_data_segmentation(args):
  # don't need to differentiate between mapping assign and test because
  # they're the same for fully unsupervised setting

  if "Coco" in args.dataset:
    dataset, dataloader = \
      _create_loader(args, cocostuff.__dict__[args.dataset],
                     partitions=args.train_partitions, purpose="train")

    test_dataset, test_dataloader = \
      _create_loader(args, cocostuff.__dict__[args.dataset],
                     partitions=args.test_partitions, purpose="test")

  elif args.dataset == "Potsdam":
    dataset, dataloader = \
      _create_loader(args, potsdam.__dict__[args.dataset],
                     partitions=args.train_partitions, purpose="train")

    test_dataset, test_dataloader = \
      _create_loader(args, potsdam.__dict__[args.dataset],
                     partitions=args.test_partitions, purpose="test")

  else:
    assert(False)

  return dataset, dataloader, test_dataset, test_dataloader

def _create_loader(args, dataset_class, partitions, purpose):
  imgs_list = []
  for partition in partitions:
    imgs_curr = dataset_class(
      **{"config": args,
      "split": partition,
      "purpose": purpose}  # return testing tuples, image and label
    )
    imgs_list.append(imgs_curr)

  imgs_dataset = ConcatDataset(imgs_list)
  dataloader = torch.utils.data.DataLoader(imgs_dataset,
                                batch_size=args.batch_sz, # full batch
                                shuffle=False, # important
                                num_workers=0,
                                drop_last=False)
  return imgs_dataset, dataloader