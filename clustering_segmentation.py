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
    tup = self.base_dataset[index]
    if len(tup) == 2: # train
      imgs, masks = tup
      return (imgs, masks, torch.from_numpy(self.pseudolabels[index]).cuda())
    else:
      assert(len(tup) == 3) # test
      imgs, labels, masks = tup
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

  """
  res = faiss.StandardGpuResources()
  flat_config = faiss.GpuIndexFlatConfig()
  flat_config.useFloat16 = False
  flat_config.device = 0
  index = faiss.GpuIndexFlatL2(res, d, flat_config)
  
  """
  if args.verbose:
    print("number of GPUs %s" % faiss.get_num_gpus())

  """

  cpu_index = faiss.IndexFlatL2(d)
  index = faiss.index_cpu_to_all_gpus(
    cpu_index
  )
  """
  res = faiss.StandardGpuResources()
  cpu_index = faiss.IndexFlatL2(d)
  index = faiss.index_cpu_to_gpu(res, 1, cpu_index)

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
    if (verbose and i < 10) or (i % int(len(dataloader) / 10) == 0):
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

    if verbose and i < 2:
      print("(run_kmeans) through sobel %d time %s" % (i, datetime.now()))
      sysout.flush()

    with torch.no_grad():
      # penultimate = features
      x_out = model(imgs, penultimate=True).cpu().numpy()

    if verbose and i < 2:
      print("(run_kmeans) through net %d time %s" % (i, datetime.now()))
      sysout.flush()

    bn, dlen, h, w = x_out.shape
    x_out = x_out.transpose((0, 2, 3, 1))
    x_out = x_out.reshape(bn * h * w, dlen)

    if pca_mat is not None:
      x_out = apply_learned_preprocessing(x_out, pca_mat)

    if verbose and i < 2:
      print("(run_kmeans) processed feat %d time %s" % (i, datetime.now()))
      sysout.flush()

    _, I = index.search(x_out, 1)

    if verbose and i < 2:
      print("(run_kmeans) index searched %d time %s" % (i, datetime.now()))
      print(I.__class__)
      if isinstance(I, np.ndarray):
        print(I.shape)
      sysout.flush()

    pseudolabels_curr = np.array([int(n[0]) for n in I], dtype=np.int32)

    if verbose and i < 2:
      print("(run_kmeans) results obtained %d time %s" % (i, datetime.now()))
      sysout.flush()

    pseudolabels_curr = pseudolabels_curr.reshape(bn, h, w)
    pseudolabels[num_imgs_curr: num_imgs_curr + bn, :, :] = pseudolabels_curr
    num_imgs_curr += bn

    if verbose and i < 2:
      print("(run_kmeans) stored %d time %s" % (i, datetime.now()))
      sysout.flush()

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

def preprocess_features_pytorch(npdata):
  # https://stats.stackexchange.com/questions/95806/how-to-whiten-the-data-using-principal-component-analysis
  d = torch.from_numpy(npdata).cuda()
  # dlen, n
  d = d.permute(0, 1)
  dlen, n = d.shape
  # demean, remove average data point
  d = d - d.mean(dim=1, keepdim=True)

  cov = d.mm(d.t())
  cov = (cov + cov.t()) / (2 * n)

  # dlen, dlen in both cases
  eig_vals, eig_vecs = torch.symeig(cov, eigenvectors=True)
  print(eig_vals.shape)
  print(eig_vecs.shape)
  assert(eig_vals.shape == (dlen, dlen))
  assert(eig_vecs.shape == (dlen, dlen))

  projected = apply_whiten_and_norm_pytorch(d, eig_vals, eig_vecs)

  return projected, eig_vals, eig_vecs

def apply_whiten_and_norm_pytorch(d, eig_vals, eig_vecs):
  dlen, n = d.shape

  # pre-apply, eig_vecs is transposed already (row format)
  # dlen, n
  projected = eig_vecs.mm(d)
  assert(projected.shape == (dlen, n))

  # scale eigenvalues
  # dlen, n
  projected = eig_vals.pow(-0.5).mm(projected)
  assert(projected.shape == (dlen, n))

  # sort and reorder the eigenvalues
  _, inds = torch.sort(torch.diagonal(eig_vals, 0))
  projected = projected[inds, :]

  # take the top ones
  smaller_dlen = int(dlen / 4)
  projected = projected[:-smaller_dlen, :]

  # revert back to row order
  projected = projected.t()
  assert(projected.shape == (n, smaller_dlen))

  # finally, l2 norm
  norms = torch.norm(projected, p=2, dim=1, keepdim=True)
  projected /= norms
  assert(projected.shape == (n, smaller_dlen))

  return projected

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
