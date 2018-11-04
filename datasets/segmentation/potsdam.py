from __future__ import print_function

import os.path as osp
import pickle
import sys
from datetime import datetime

import os
import cv2
import numpy as np
import scipy.io as sio
import torch
import torchvision.transforms as tvt
from PIL import Image
from torch.utils import data
from tqdm import tqdm

from utils.segmentation.render import render
from utils.segmentation.transforms import \
  pad_and_or_crop, random_affine, random_translation, custom_greyscale_numpy

__all__ = ["Potsdam"]

CHECK_TIME = False
CHECK_TEST_DATA = False
CHECK_TRAIN_DATA = False

class _Potsdam(data.Dataset):
  """Base class
  This contains fields and methods common to all Potsdam datasets:
  PotsdamFull (6)
  PotsdamFew (3)

  """

  def __init__(self, config=None, split=None, purpose=None, preload=False):
    super(_Potsdam, self).__init__()

    self.split = split
    self.purpose = purpose

    self.root = config.dataset_root

    assert(os.path.exists(os.path.join(self.root, "debugged.out")))

    # always used (labels fields used to make relevancy mask for train)
    self.gt_k = config.gt_k
    self.pre_scale_all = config.pre_scale_all
    self.pre_scale_factor = config.pre_scale_factor
    self.input_sz = config.input_sz

    self.do_rgb = config.do_rgb
    self.do_sobel = config.do_sobel

    # only used if purpose is train
    if purpose == "train":
      self.jitter_tf = tvt.ColorJitter(brightness=config.jitter_brightness,
                                       contrast=config.jitter_contrast,
                                       saturation=config.jitter_saturation,
                                       hue=config.jitter_hue)

      self.flip_p = config.flip_p  # 0.5

    self.preload = preload

    self.files = []
    self.images = []
    self.labels = []

    self._set_files()

    if self.preload:
      self._preload_data()

    cv2.setNumThreads(0)

  def _set_files(self):
    raise NotImplementedError()

  def _load_data(self, image_id):
    raise NotImplementedError()

  def _prepare_train(self, index, img):
    # This returns gpu tensors.
    # label is passed in canonical [0 ... 181] indexing

    img = img.astype(np.float32)

    # shrink original images, for memory purposes
    # or enlarge
    if self.pre_scale_all:
      img = cv2.resize(img, dsize=None, fx=self.pre_scale_factor,
                       fy=self.pre_scale_factor,
                       interpolation=cv2.INTER_LINEAR)

    # random crop to input sz
    img, coords = pad_and_or_crop(img, self.input_sz, mode="random")

    # more data augmentation: flip and jitter
    # images are RGBIR. We don't want to jitter or greyscale the IR part
    img_ir = img[:, :, 3]
    img = img[:, :, :3]

    img = Image.fromarray(img.astype(np.uint8))

    img = self.jitter_tf(img)  # not in place, new memory
    img = np.array(img)

    # get greyscale if doing sobel, keep rgb if needed
    # channels still last
    if self.do_sobel:
      img = custom_greyscale_numpy(img, include_rgb=self.do_rgb)

    img = img.astype(np.float32) / 255.

    # concatenate IR back on before spatial warps
    # may be concatenating onto just greyscale image
    # grey/RGB underneath IR
    img_ir = img_ir.astype(np.float32) / 255.
    img = np.concatenate([img, np.expand_dims(img_ir, axis=2)], axis=2)

    # convert to channel-first tensor format
    # make them all cuda tensors now, except label, for optimality
    img = torch.from_numpy(img).permute(2, 0, 1).cuda()

    # (img2) do random flip, tf_mat changes
    if np.random.rand() > self.flip_p:
      img = torch.flip(img, dims=[2])  # horizontal, along width

    # uint8 tensor as masks should be binary, also for consistency,
    # but converted to float32 in main loop because is used
    # multiplicatively in loss
    mask_img = torch.ones(self.input_sz, self.input_sz).to(torch.uint8).cuda()

    if CHECK_TRAIN_DATA:
      render(img, mode="image", name=("train_data_img_%d" % index))
      render(mask_img, mode="mask", name=("train_data_mask_%d" % index))

    return img, mask_img

  def _prepare_test(self, index, img, label):
    # This returns cpu tensors.
    #   Image: 3D with channels last, float32, in range [0, 1] (normally done
    #     by ToTensor).
    #   Label map: 2D, flat int64, [0 ... sef.gt_k - 1]
    # label is passed in canonical [0 ... 181] indexing

    assert(label is not None)

    assert (img.shape[:2] == label.shape)
    img = img.astype(np.float32)
    label = label.astype(np.int32)

    # shrink original images, for memory purposes, or magnify
    if self.pre_scale_all:
      img = cv2.resize(img, dsize=None, fx=self.pre_scale_factor,
                       fy=self.pre_scale_factor,
                       interpolation=cv2.INTER_LINEAR)
      label = cv2.resize(label, dsize=None, fx=self.pre_scale_factor,
                         fy=self.pre_scale_factor,
                         interpolation=cv2.INTER_NEAREST)

    # center crop to input sz
    img, _ = pad_and_or_crop(img, self.input_sz, mode="centre")
    label, _ = pad_and_or_crop(label, self.input_sz, mode="centre")

    img_ir = img[:, :, 3]
    img = img[:, :, :3]

    # finish
    # may be concatenating onto just greyscale image
    if self.do_sobel:
      img = custom_greyscale_numpy(img, include_rgb=self.include_rgb)

    img = img.astype(np.float32) / 255.

    img_ir = img_ir.astype(np.float32) / 255.
    img = np.concatenate([img, np.expand_dims(img_ir, axis=2)], axis=2) #
    # grey/RGB under IR

    img = torch.from_numpy(img).permute(2, 0, 1)

    if CHECK_TEST_DATA:
      render(label, mode="label", name=("test_data_label_pre_%d" % index))

    # convert to coarse if required, reindex to [0, gt_k -1], and get mask
    label = self._filter_label(label)
    mask = torch.ones(self.input_sz, self.input_sz).to(torch.uint8)

    if CHECK_TEST_DATA:
      render(img, mode="image", name=("test_data_img_%d" % index))
      render(label, mode="label", name=("test_data_label_post_%d" % index))
      render(mask, mode="mask", name=("test_data_mask_%d" % index))

    # dataloader must return tensors (conversion forced in their code anyway)
    return img, torch.from_numpy(label), mask

  def _preload_data(self):
    for image_id in tqdm(
      self.files, desc="Preloading...", leave=False, dynamic_ncols=True):
      image, label = self._load_data(image_id)
      self.images.append(image)
      self.labels.append(label)

  def __getitem__(self, index):
    if self.preload:
      image, label = self.images[index], self.labels[index]
    else:
      image_id = self.files[index]
      image, label = self._load_data(image_id)

    if self.purpose == "train":
      return self._prepare_train(index, image)
    else:
      assert (self.purpose == "test")
      return self._prepare_test(index, image, label)

  def __len__(self):
    return len(self.files)

  def _check_gt_k(self):
    raise NotImplementedError()

  def _filter_label(self, label):
    raise NotImplementedError()

  def _set_files(self):
    if self.split in ["unlabelled_train", "labelled_train", "labelled_test"]:
      # deterministic order - important - so >1 dataloader actually meaningful
      file_list = osp.join(self.root, self.split + ".txt")
      file_list = tuple(open(file_list, "r"))
      file_list = [id_.rstrip() for id_ in file_list]
      self.files = file_list # list of ids which may or may not have gt
    else:
      raise ValueError("Invalid split name: {}".format(self.split))

  def _load_data(self, image_id):
    image_path = osp.join(self.root, "imgs", image_id + ".mat")
    label_path = osp.join(self.root, "gt", image_id + ".mat")

    image = sio.loadmat(image_path)["img"]
    assert(image.dtype == np.uint8)

    if os.path.exists(label_path):
      label = sio.loadmat(label_path)["gt"]
      assert(label.dtype == np.int32)
      return image, label
    else:
      return image, None

class Potsdam(_Potsdam):
  def __init__(self, **kwargs):
    super(Potsdam, self).__init__(**kwargs)

    config = kwargs["config"]
    self.use_coarse_labels = config.use_coarse_labels
    self._check_gt_k()

    # see potsdam_prepare.py
    self._fine_to_coarse_dict = {0: 0, 4: 0, # roads and cars
                                 1: 1, 5: 1, # buildings and clutter
                                 2: 2, 3: 2 # vegetation and trees
                                 }

  def _check_gt_k(self):
    if self.use_coarse_labels:
      assert(self.gt_k == 3)
    else:
      assert(self.gt_k == 6)

  def _filter_label(self, label):
    if self.use_coarse_labels:
      new_label_map = np.zeros(label.shape, dtype=label.dtype)

      for c in xrange(6):
        new_label_map[label == c] = self._fine_to_coarse_dict[c]

      return new_label_map
    else:
      assert(label.max() < self.gt_k)
      return label


