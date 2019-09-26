import cv2
import numpy as np
import math
import scipy.signal


# preprocesses given image, returns padded version and array w/ original dimensions for final image
def prepare_image(im: np.ndarray, padding: int):
    dimensions = im.shape
    # convert brg to grayscale
    if (len(dimensions)) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # zero pad unpadded image
    im2 = np.zeros((dimensions[0], dimensions[1]))
    im = np.pad(im, padding, 'constant', constant_values=0)
    # ensure image array is large enough to prevent overflow
    return im.astype(dtype='int64'), im2  # return processed image


# post processes given image
def pp_image(im: np.ndarray, g2rgb: bool) -> np.ndarray:
    if g2rgb:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    # return pp image with correct type
    return im.astype(dtype='uint8')


# unused function
# designed to get neighbor pixels for given location and kernel size
# untested, may not work correctly
def get_pixel_neighbors(im: np.ndarray, x: int, y: int, kernel: int) -> np.ndarray:
    array = np.zeros((2 * kernel + 1, 2 * kernel + 1))
    total = 0
    for i in range(0, 2 * kernel + 1):
        for j in range(0, 2 * kernel + 1):
            array[i, j] = im[x - kernel + i, y - kernel + j]

    return array
