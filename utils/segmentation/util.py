import torch
import numpy as np

def compute_spatial_features(args, dataloader, model, num_imgs):
  model.eval()
  # discard the label information in the dataloader
  for i, tup in enumerate(dataloader):
    if len(tup) == 3: # test dataset, cpu
      imgs, _, mask = tup
      imgs = imgs.cuda()
    else: # cuda
      assert(len(tup) == 2)
      imgs, mask = tup
      mask = mask.cpu()

    mask = mask.numpy().astype(np.bool)

    with torch.no_grad():
      # penultimate = features
      x_out = model(imgs, penultimate=True).cpu().numpy().astype(np.float32)

    if i == 0:
      assert(x_out.shape[1] == model.module.dlen)
      assert(x_out.shape[2] == args.input_sz and
             x_out.shape[3] == args.input_sz)
      print("%d %d %d" % (num_imgs, x_out.shape[1], args.input_sz))
      features = np.zeros((num_imgs, x_out.shape[1], args.input_sz,
                           args.input_sz), dtype=np.float32)
      masks = np.zeros((num_imgs, args.input_sz, args.input_sz), dtype=np.bool)

    if i < len(dataloader) - 1:
      features[i * args.batch_sz: (i + 1) * args.batch_sz] = x_out
      masks[i * args.batch_sz: (i + 1) * args.batch_sz] = mask
    else:
      features[i * args.batch_sz:] = x_out
      masks[i * args.batch_sz:] = mask

  return features, masks