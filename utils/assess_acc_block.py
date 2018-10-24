import torch
import numpy as np

def assess_acc_block(net, test_loader, gt_k, contiguous_sz):
  total = 0
  all = None
  all_targets = None

  net.eval()
  for i, (imgs, targets) in enumerate(test_loader):
    with torch.no_grad():
      x_out = net(imgs.cuda(), penultimate=True)

    bn, dlen = x_out.shape
    if all is None:
      all = np.zeros((len(test_loader) * bn, dlen))
      all_targets = np.zeros(len(test_loader) * bn)

    all[total:(total + bn), :] = x_out.cpu().numpy()
    all_targets[total:(total + bn)] = targets.numpy()
    total += bn

  # 40000
  all = all[:total, :]
  all_targets = all_targets[:total]

  num_orig, leftover = divmod(total, contiguous_sz)
  assert(leftover == 0)

  all = all.reshape((num_orig, contiguous_sz, dlen))
  all = all.sum(axis=1, keepdims=False) / float(contiguous_sz)

  all_targets = all_targets.reshape((num_orig, contiguous_sz))
  # sanity check
  all_targets_avg = all_targets.astype("int").sum(axis=1)/ contiguous_sz
  all_targets = all_targets[:, 0].astype("int")
  assert(np.array_equal(all_targets_avg, all_targets))

  preds = np.argmax(all, axis=1).astype("int")
  assert (preds.min() >= 0 and preds.max() < gt_k)
  assert (all_targets.min() >= 0 and all_targets.max() < gt_k)
  if not (preds.shape == all_targets.shape):
    print(preds.shape)
    print(all_targets.shape)
    exit(1)

  assert(preds.shape == (num_orig,))
  correct = (preds == all_targets).sum()

  return correct / float(num_orig)