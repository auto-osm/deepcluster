import numpy as np

def custom_cutout(min_box=None, max_box=None):
  def _inner(img):
    w, h = img.size

    # find left, upper, right, lower
    box_sz = np.random.randint(min_box, max_box + 1)
    half_box_sz = int(np.floor(box_sz / 2.))
    x_c = np.random.randint(half_box_sz, w - half_box_sz)
    y_c = np.random.randint(half_box_sz, h - half_box_sz)
    box = (x_c - half_box_sz, y_c - half_box_sz, x_c + half_box_sz, y_c + half_box_sz)

    img.paste(0, box=box)
    return img

  return _inner