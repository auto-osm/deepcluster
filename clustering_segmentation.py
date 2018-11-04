import faiss
import numpy as np
import torch
import torch.utils.data as data
from clustering import preprocess_features
from utils.segmentation.transforms import sobel_process
from datetime import datetime
from sys import stdout as sysout

__all__ = ['Kmeans', 'cluster_assign']

class ReassignedDataset(data.Dataset):
  """A dataset where the new images labels are given in argument."""

  def __init__(self, pseudolabels, dataset):
    # recall pseudolabels contain garbage at mask locations
    self.pseudolabels = pseudolabels
    self.base_dataset = dataset

    assert(isinstance(self.pseudolabels, np.ndarray))
    assert(self.pseudolabels.shape[0] == len(self.base_dataset))

  def __getitem__(self, index):
    if self.base_dataset.purpose == "train":
      imgs, masks = self.base_dataset[index]
      return (imgs, masks, torch.from_numpy(self.pseudolabels[index]).cuda())
    else:
      assert(self.base_dataset.purpose == "test")
      imgs, labels, masks = self.base_dataset[index]
      return (imgs, masks, torch.from_numpy(self.pseudolabels[index]).cuda(),
              labels)

  def __len__(self):
    return len(self.base_dataset)

def cluster_assign(pseudolabels, dataset):
  """Creates a dataset from clustering, with clusters as labels.
  """
  return ReassignedDataset(pseudolabels, dataset)

def run_kmeans(args, unmasked_vectorised_feat, nmb_clusters, dataloader,
               num_imgs, model, pca_mat,
               verbose=False):
  """Runs kmeans on 1 GPU.
  Args:
      x: data
      nmb_clusters (int): number of clusters
  Returns:
      list: ids of data in each cluster
  """

  # d is not dlen - dimensionality reduced!
  n_data, d = unmasked_vectorised_feat.shape

  if verbose:
    print("starting cluster in run_kmeans %s" % datetime.now())

  # faiss implementation of k-means
  clus = faiss.Clustering(d, nmb_clusters)
  clus.niter = 20
  clus.max_points_per_centroid = 100000000
  res = faiss.StandardGpuResources()
  flat_config = faiss.GpuIndexFlatConfig()
  flat_config.useFloat16 = False
  flat_config.device = 0
  index = faiss.GpuIndexFlatL2(res, d, flat_config)

  # perform the training
  clus.train(unmasked_vectorised_feat, index)

  losses = faiss.vector_to_array(clus.obj)
  centroids = faiss.vector_to_array(clus.centroids).reshape(clus.k, clus.d)

  if verbose:
    print("trained cluster, starting pseudolabel collection %s" %
          datetime.now())
    sysout.flush()

  # perform inference on spatially preserved features (incl masked)
  num_imgs_curr = 0
  for i, tup in enumerate(dataloader):
    if verbose and i < 10:
      print("(run_kmeans) batch %d time %s" % (i, datetime.now()))
      sysout.flush()

    if len(tup) == 3: # test dataset, cuda
      imgs, _, _ = tup
    else: # cuda
      assert(len(tup) == 2)
      imgs, _ = tup

    if i == 0:
      pseudolabels = np.zeros((num_imgs, args.input_sz, args.input_sz),
                              dtype=np.int32)

    if args.do_sobel:
      imgs = sobel_process(imgs, args.do_rgb, using_IR=args.using_IR)
      # now rgb(ir) and/or sobel

    assert(imgs.is_cuda)

    with torch.no_grad():
      # penultimate = features
      x_out = model(imgs, penultimate=True)

    bn, dlen, h, w = x_out.shape
    x_out = x_out.transpose((0, 2, 3, 1))
    x_out = x_out.reshape(bn * h * w, dlen).cpu().numpy().astype(np.float32)

    if pca_mat is not None:
      x_out = apply_learned_preprocessing(x_out, pca_mat)

    _, I = index.search(x_out, 1)
    pseudolabels_curr = np.array([int(n[0]) for n in I], dtype=np.int32)
    pseudolabels_curr = pseudolabels_curr.reshape(bn, h, w)

    pseudolabels[num_imgs_curr: num_imgs_curr + bn, :, :] = pseudolabels_curr
    num_imgs_curr += bn

  assert(num_imgs == num_imgs_curr)

  return pseudolabels, losses[-1], centroids

def apply_learned_preprocessing(npdata, mat):
  assert(len(npdata.shape) == 2)
  assert(mat.is_trained)

  npdata = mat.apply_py(npdata)

  # L2 normalization
  row_sums = np.linalg.norm(npdata, axis=1)
  npdata = npdata / row_sums[:, np.newaxis]

  return npdata

class Kmeans:
  def __init__(self, k):
    self.k = k

  def cluster(self, args, features, dataloader, num_imgs, model,
              proc_feat=False,
              verbose=False):
    """Performs k-means clustering.
        Args:
            x_data (np.array N * dim): data to cluster
    """

    # already vectorised
    # need to use pca_mat here unlike in clustering, because inference data
    # != training data for the clusterer
    if proc_feat:
      features, pca_mat = preprocess_features(features)
    else:
      pca_mat = None

    # cluster the features and perform inference on spatially uncollapsed
    # dataset

    pseudolabelled_x, loss, centroids = run_kmeans(args, features,
                                                   self.k,
                                                   dataloader, num_imgs,
                                                   model,
                                                   pca_mat,
                                                   verbose)

    # no need to store masks, reloaded in dataloader later
    self.centroids = centroids
    self.pseudolabelled_x = pseudolabelled_x

    return loss
