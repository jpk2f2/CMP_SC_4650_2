import numpy as np
import scipy.signal
import math
import matplotlib as plot
import utility
import smoothing
import masks
import cv2


def guass_pyramid_resize(im: np.ndarray) -> np.ndarray:
    im = smoothing.guass_filter(im, 1, False)
    dimensions = im.shape
    im_ds = np.zeros((dimensions[0]//2, dimensions[1]//2))
    dimensions = im_ds.shape
    for i in range(0, dimensions[0]):
        for j in range(0, dimensions[1]):
            im_ds[i, j] = im[2 * i, 2 * j]

    # cv2.imshow('test', im_ds)
    im_ds = utility.pp_image(im_ds, False)

    return im_ds
