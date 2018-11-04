from utils.segmentation.data import make_data_segmentation
from utils.segmentation.util import compute_vectorised_features
import clustering_segmentation
import numpy as np

from sklearn.utils.linear_assignment_ import linear_assignment
from ..clustering.assess_acc import analyse, compute_acc
from datetime import datetime
from sys import stdout as sysout

TIME = True

def assess_acc_segmentation(args, test_dataset, test_dataloader, model,
                            num_imgs):
  # use gt_k here, unlike in training when using k
  deepcluster = clustering_segmentation.__dict__[args.clustering](args.gt_k)

  # n, h, w
  if args.verbose:
    print("starting features %s" % datetime.now())
    sysout.flush()
  features = compute_vectorised_features(args, test_dataloader, model,
                                                num_imgs)

  if args.verbose:
    print("starting cluster %s" % datetime.now())
    sysout.flush()

  assess_cluster_loss = deepcluster.cluster(args, features, test_dataloader,
                                            len(test_dataset), model,
                                        proc_feat=args.proc_feat,
                                        verbose=args.verbose)

  if args.verbose:
    print("gotten pseudolabels %s" % datetime.now())
    sysout.flush()

  test_dataset = clustering_segmentation.cluster_assign(
    deepcluster.pseudolabelled_x,
    test_dataset)

  if args.verbose:
    print("gotten new dataset %s" % datetime.now())
    sysout.flush()

  # maxed
  vectorised_unmasked_preds = np.zeros((num_imgs * args.input_sz *
                                        args.input_sz), dtype=np.int32)
  vectorised_unmasked_targets = np.zeros((num_imgs * args.input_sz *
                                          args.input_sz), dtype=np.int32)
  actual_num_samples = 0

  for _, masks, preds, targets in test_dataset: # cuda, imgs one by one...
    curr_num_samples = masks.sum()

    preds = preds.masked_select(masks).cpu().numpy()
    targets = targets.masked_select(masks).cpu().numpy()

    vectorised_unmasked_preds[actual_num_samples:actual_num_samples +
                                                 curr_num_samples] = preds
    vectorised_unmasked_targets[actual_num_samples:actual_num_samples +
                                                 curr_num_samples] = targets

    actual_num_samples += curr_num_samples

  predicted_labels = vectorised_unmasked_preds[:actual_num_samples]
  true_labels = vectorised_unmasked_targets[:actual_num_samples]

  if args.verbose:
    print("gotten unmasked preds (pseudolabels) and targets %s" %
          datetime.now())
    sysout.flush()

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

  if args.verbose:
    print("doing analyse %s" % datetime.now())
    sysout.flush()

  distribution, centroid_min_max = analyse(reordered_preds, args.gt_k,
                                           deepcluster.centroids)

  if args.verbose:
    print("doing acc %s" % datetime.now())
    sysout.flush()

  acc = compute_acc(reordered_preds, true_labels, args.gt_k)

  if args.verbose:
    print("finished assess_acc %s" % datetime.now())
    sysout.flush()

  return acc, distribution, centroid_min_max, assess_cluster_loss
