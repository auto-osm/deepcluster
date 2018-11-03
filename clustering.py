# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import faiss
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['PIC', 'Kmeans', 'cluster_assign', 'arrange_clustering', 'preprocess_features']


def pil_loader(path):
  """Loads an image.
  Args:
      path (string): path to image file
  Returns:
      Image
  """
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


class ReassignedDataset(data.Dataset):
  """A dataset where the new images labels are given in argument.
  Args:
      image_indexes (list): list of image indexes in the dataset
      pseudolabels (list): list of labels for each image
                           lines up with image_indexes
      dataset (list): list of tuples with paths to images
      transform (callable, optional): a function/transform that takes in
                                      an PIL image and returns a
                                      transformed version
  """

  def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
    self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
    self.transform = transform

  def make_dataset(self, shuffled_image_indexes, pseudolabels, dataset):
    # passed in: list of image names (index in dataset) and list of
    # pseudolabels

    # pseudolabels is chunked 0...0, 1...1 etc
    # attempt to reindex the pseudolabels for no reason?
    # label_to_idx = {label: idx for idx, label in enumerate(set(
    # pseudolabels))}

    # make images in original order of dataset so that it's lined up for
    # assess
    images = [None for _ in xrange(len(shuffled_image_indexes))]
    for j, idx in enumerate(shuffled_image_indexes):
      img = dataset[idx][0]  # path or image, either way, identifier
      pseudolabel = pseudolabels[j]
      # images.append((img, pseudolabel))
      images[idx] = (img, pseudolabel)
    # print("images")
    # print(images[:50])
    return images

  def __getitem__(self, index):
    """
    Args:
        index (int): index of data
    Returns:
        tuple: (image, pseudolabel) where pseudolabel is the cluster of index 
        datapoint
    """
    path, pseudolabel = self.imgs[index]

    """
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel
    """
    return path, pseudolabel

  def __len__(self):
    return len(self.imgs)


def preprocess_features(npdata):
  """Preprocess an array of features.
  Args:
      npdata (np.array N * ndim): features to preprocess
      pca (int): dim of output
  Returns:
      np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
  """
  _, ndim = npdata.shape
  npdata = npdata.astype('float32')

  # quarter the dimensions
  pca = int(ndim / 4.)

  # Apply PCA-whitening with Faiss
  mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
  mat.train(npdata)
  assert mat.is_trained
  npdata = mat.apply_py(npdata)

  # L2 normalization
  row_sums = np.linalg.norm(npdata, axis=1)
  npdata = npdata / row_sums[:, np.newaxis]

  return npdata


def make_graph(xb, nnn):
  """Builds a graph of nearest neighbors.
  Args:
      xb (np.array): data
      nnn (int): number of nearest neighbors
  Returns:
      list: for each data the list of ids to its nnn nearest neighbors
      list: for each data the list of distances to its nnn NN
  """
  N, dim = xb.shape

  # we need only a StandardGpuResources per GPU
  res = faiss.StandardGpuResources()

  # L2
  flat_config = faiss.GpuIndexFlatConfig()
  flat_config.device = int(torch.cuda.device_count()) - 1
  index = faiss.GpuIndexFlatL2(res, dim, flat_config)
  index.add(xb)
  D, I = index.search(xb, nnn + 1)
  return I, D


def cluster_assign(args, images_lists, dataset, tra=None):
  """Creates a dataset from clustering, with clusters as labels.
  Args:
      images_lists (list of list): for each cluster, the list of image indexes
                                  belonging to this cluster
      dataset (list): initial dataset
  Returns:
      ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                   labels
  """
  assert images_lists is not None
  pseudolabels = []
  image_indexes = []
  for cluster, images in enumerate(images_lists):
    image_indexes.extend(images)
    pseudolabels.extend([cluster] * len(images))

  return ReassignedDataset(image_indexes, pseudolabels, dataset, tra)


def run_kmeans(x, nmb_clusters, verbose=False):
  """Runs kmeans on 1 GPU.
  Args:
      x: data
      nmb_clusters (int): number of clusters
  Returns:
      list: ids of data in each cluster
  """
  n_data, d = x.shape

  # faiss implementation of k-means
  clus = faiss.Clustering(d, nmb_clusters)
  clus.niter = 20
  clus.max_points_per_centroid = 10000000
  res = faiss.StandardGpuResources()
  flat_config = faiss.GpuIndexFlatConfig()
  flat_config.useFloat16 = False
  flat_config.device = 0
  index = faiss.GpuIndexFlatL2(res, d, flat_config)

  # perform the training
  clus.train(x, index)
  _, I = index.search(x, 1)
  losses = faiss.vector_to_array(clus.obj)

  centroids = faiss.vector_to_array(clus.centroids).reshape(clus.k, clus.d)
  # if verbose: print('k-means loss evolution: {0}'.format(losses))

  return [int(n[0]) for n in I], losses[-1], centroids


def arrange_clustering(images_lists):
  pseudolabels = []
  image_indexes = []
  for cluster, images in enumerate(images_lists):
    image_indexes.extend(images)
    pseudolabels.extend([cluster] * len(images))
  indexes = np.argsort(image_indexes)
  return np.asarray(pseudolabels)[indexes]


class Kmeans:
  def __init__(self, k):
    self.k = k

  def cluster(self, data, proc_feat=False, verbose=False):
    """Performs k-means clustering.
        Args:
            x_data (np.array N * dim): data to cluster
    """

    # PCA-reducing, whitening and L2-normalization
    if proc_feat:
      data = preprocess_features(data)

    # cluster the data
    # I: data index -> k means cluster index
    # images_lists: k means cluster index -> data index
    # OLD:
    I, loss, centroids = run_kmeans(data, self.k, verbose)

    # I, loss, centroids = run_our_kmeans(data, self.k)

    self.centroids = centroids

    # maps cluster index to
    self.images_lists = [[] for i in range(self.k)]
    for i in range(len(data)):
      self.images_lists[I[i]].append(i)

    # if verbose:
    #    print('k-means time: {0:.0f} s'.format(time.time() - end))

    return loss
