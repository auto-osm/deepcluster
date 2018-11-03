import faiss
import numpy as np
import torch
import torch.utils.data as data
from clustering import preprocess_features

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

def run_kmeans(unmasked_vectorised_feat, nmb_clusters, x, verbose=False):
  """Runs kmeans on 1 GPU.
  Args:
      x: data
      nmb_clusters (int): number of clusters
  Returns:
      list: ids of data in each cluster
  """
  n_data, d = unmasked_vectorised_feat.shape

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
  clus.train(unmasked_vectorised_feat, index)

  # perform inference on spatially preserved features
  # doesn't matter that masked pixels are still included in x
  n, h, w, d2 = x.shape
  assert(n == n_data and d2 == d)
  x = x.reshape(n * h * w, d2)
  _, I = index.search(x, 1)
  pseudolabels = np.array([int(n[0]) for n in I], dtype=np.int32)
  pseudolabels = pseudolabels.reshape(n, h, w)

  losses = faiss.vector_to_array(clus.obj)
  centroids = faiss.vector_to_array(clus.centroids).reshape(clus.k, clus.d)

  return pseudolabels, losses[-1], centroids

class Kmeans:
  def __init__(self, k):
    self.k = k

  def cluster(self, x_out, masks, proc_feat=False, verbose=False):
    """Performs k-means clustering.
        Args:
            x_data (np.array N * dim): data to cluster
    """

    # get unmasked and vectorised features for training
    n, d, h, w = x_out.shape
    assert (masks.shape == (n, h, w))
    assert (masks.dtype == np.bool)

    x = x_out.transpose((0, 2, 3, 1))  # features last
    unmasked_vectorised_feat = x[masks, :]
    num_unmasked = unmasked_vectorised_feat.shape[0]
    assert (num_unmasked == masks.sum())
    unmasked_vectorised_feat = unmasked_vectorised_feat.reshape(num_unmasked, d)

    # PCA-reducing, whitening and L2-normalization
    if proc_feat:
      unmasked_vectorised_feat = preprocess_features(unmasked_vectorised_feat)

    # cluster the features and perform inference on spatially uncollapsed x
    pseudolabelled_x, loss, centroids = run_kmeans(unmasked_vectorised_feat,
                                                   self.k, x, verbose)

    # no need to store masks, reloaded in dataloader later
    self.centroids = centroids
    self.pseudolabelled_x = pseudolabelled_x

    return loss
