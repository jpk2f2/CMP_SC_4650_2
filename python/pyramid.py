import numpy as np
import scipy.signal
import math
import matplotlib as plot
import utility
import smoothing
import masks
import cv2
from typing import List



def guass_pyramid_resize(im: np.ndarray) -> np.ndarray:
    im = smoothing.guass_filter(im, 1, False)
    dimensions = im.shape
    im_ds = np.zeros((dimensions[0]//2, dimensions[1]//2))
    dimensions = im_ds.shape
    for i in range(0, dimensions[0]):
        for j in range(0, dimensions[1]):
            im_ds[i, j] = im[2 * i, 2 * j]

    im_ds = utility.pp_image(im_ds, False)

    return im_ds


def get_gaussian_pyramid(im: np.ndarray, levels: int) -> List[np.ndarray]:
    if levels > 7:
        levels = 7
    if levels <= 1:
        return [im, ]

    x = get_gaussian_pyramid(im, levels-1)
    x.append(guass_pyramid_resize(x[-1]))
    return x



