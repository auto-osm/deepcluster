import clustering
from util import compute_features
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

def assess_acc(args, test_dataset, test_dataloader, model, num_imgs):
  # new clusterer
  deepcluster = clustering.__dict__[args.clustering](args.gt_k)
  features = compute_features(args, test_dataloader, model, num_imgs)

  assess_cluster_loss = deepcluster.cluster(features,
                                            proc_feat=args.proc_feat,
                                            verbose=args.verbose)

  # print("images_list sizes of clusterer after cluster")
  # for i in xrange(len(deepcluster.images_lists)):
  #    print("gt_k: %d (%d)" % (i, len(deepcluster.images_lists[i])))

  # non shuffled
  relabelled_test_dataset = clustering.cluster_assign(args,
                                                      deepcluster.images_lists,
                                                      test_dataset)

  assert (num_imgs == len(test_dataset))
  assert (num_imgs == len(relabelled_test_dataset))

  # non shuffled
  true_labels = np.array([test_dataset[i][1] for i in xrange(num_imgs)])

  predicted_labels = np.array(
    [relabelled_test_dataset[i][1] for i in xrange(num_imgs)])

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


def analyse(predictions, gt_k, centroids):
  # bar chart showing assignment per cluster centre (named)

  predictions = np.array(predictions)
  sums = np.array([sum(predictions == c) for c in xrange(gt_k)])
  assert (len(predictions) == sum(sums))

  sizes = get_sizes(centroids)

  return sums, (sizes.min(), sizes.max())


def get_sizes(centroids):
  # k, d matrix
  # e.g. 10, 3200 (stl10 with net5g)

  k, d = centroids.shape

  return np.abs(centroids).sum(axis=1) / float(d)


def compute_acc(preds, targets, num_k):
  assert (preds.shape == targets.shape)
  acc = 0
  for c in xrange(num_k):
    curr_acc = ((preds == c) * (targets == c)).sum()  # TP
    acc += curr_acc
  return acc / float(preds.shape[0])
