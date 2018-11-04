import torch
import numpy as np
from utils.segmentation.transforms import sobel_process
from datetime import datetime
from sys import stdout as sysout

def compute_vectorised_features(args, dataloader, model, num_imgs):

  max_num_pixels_per_img = int(args.max_num_pixel_samples / num_imgs)

  features = np.zeros((args.max_num_pixel_samples, model.module.dlen),
                      dtype=np.float32)
  actual_num_features = 0

  model.eval()
  # discard the label information in the dataloader
  for i, tup in enumerate(dataloader):
    if args.verbose and i < 10:
      print("(compute_vectorised_features) batch %d time %s" % (i,
                                                                datetime.now()))
      sysout.flush()

    if len(tup) == 3: # test dataset, cuda
      imgs, _, mask = tup
    else: # cuda
      assert(len(tup) == 2)
      imgs, mask = tup

    num_unmasked = mask.sum()

    if args.do_sobel:
      imgs = sobel_process(imgs, args.do_rgb, using_IR=args.using_IR)
      # now rgb(ir) and/or sobel

    assert(imgs.is_cuda)
    assert(mask.is_cuda)

    with torch.no_grad():
      # penultimate = features
      x_out = model(imgs, penultimate=True) # torch cuda float32

    num_imgs_batch = x_out.shape[0]
    x_out = x_out.permute((0, 2, 3, 1))  # features last

    x_out = x_out.masked_select(mask.unsqueeze(3))

    if i == 0:
      assert(x_out.shape[1] == model.module.dlen)
      assert(len(x_out.shape) == 2)
      assert(x_out.shape[0] == num_unmasked)

    # select pixels randomly, and record how many selected
    num_selected = min(num_unmasked, num_imgs_batch * max_num_pixels_per_img)
    selected = torch.from_numpy(np.random.choice(num_selected,
                                                 replace=False)).cuda()

    x_out = x_out[selected, :]

    x_out = x_out.cpu().numpy() # lastly

    features[actual_num_features:actual_num_features + num_selected, :] = x_out
    actual_num_features += num_selected

  features = features[:actual_num_features, :]

  return features