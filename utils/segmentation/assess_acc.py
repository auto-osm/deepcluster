from utils.segmentation.data import make_data_segmentation
from utils.segmentation.util import compute_spatial_features
import clustering_segmentation
import numpy as np

from sklearn.utils.linear_assignment_ import linear_assignment
from ..clustering.assess_acc import analyse, compute_acc


def assess_acc_segmentation(args, test_dataset, test_dataloader, model,
                            num_imgs):
  deepcluster = clustering_segmentation.__dict__[args.clustering](args.k)

  # n, h, w
  features, masks = compute_spatial_features(args, test_dataloader, model,
                                             num_imgs)

  assess_cluster_loss = deepcluster.cluster(features, masks,
                                        proc_feat=args.proc_feat,
                                        verbose=args.verbose)

  test_dataset = clustering_segmentation.cluster_assign(
    deepcluster.pseudolabelled_imgs,
    test_dataset)

  # maxed
  vectorised_unmasked_preds = np.zeros((num_imgs * args.input_sz *
                                        args.input_sz), dtype=np.int32)
  vectorised_unmasked_targets = np.zeros((num_imgs * args.input_sz *
                                          args.input_sz), dtype=np.int32)
  actual_num_samples = 0

  for _, masks, preds, targets in test_dataset: # cpu tensors already
    curr_num_samples = masks.sum()

    preds = preds.masked_select(masks).numpy()
    targets = targets.masked_select(masks).numpy()

    vectorised_unmasked_preds[actual_num_samples:actual_num_samples +
                                                 curr_num_samples] = preds
    vectorised_unmasked_targets[actual_num_samples:actual_num_samples +
                                                 curr_num_samples] = targets

    actual_num_samples += curr_num_samples

  predicted_labels = vectorised_unmasked_preds[:actual_num_samples]
  true_labels = vectorised_unmasked_targets[:actual_num_samples]

  assert (true_labels.min() == 0)
  assert (true_labels.max() == args.gt_k - 1)
  assert (predicted_labels.min() >= 0)
  assert (predicted_labels.max() < args.gt_k)

  # hungarian matching
  num_correct = np.zeros((args.gt_k, args.gt_k))
  for i in xrange(num_imgs):
    num_correct[predicted_labels[i], true_labels[i]] += 1
  match = linear_assignment(num_imgs - num_correct)

  reordered_preds = np.zeros(num_imgs, dtype="int")
  for pred_i, target_i in match:
    reordered_preds[predicted_labels == pred_i] = target_i

  distribution, centroid_min_max = analyse(reordered_preds, args.gt_k,
                                           deepcluster.centroids)

  acc = compute_acc(reordered_preds, true_labels, args.gt_k)

  return acc, distribution, centroid_min_max, assess_cluster_loss
