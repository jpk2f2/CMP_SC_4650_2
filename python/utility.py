import cv2
import numpy as np
import math
import scipy.signal


# preprocesses given image, returns padded version and array w/ original dimensions for final image
# takes in image, desired padding, and pad type such as 'zero' or 'repeat'
def prepare_image(im: np.ndarray, padding: int, pad_type: str):
    dimensions = im.shape  # get dimensions of original image for creating new image
    # convert brg to grayscale
    if (len(dimensions)) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # create new image w/ appropriate dimensions
    im2 = np.zeros((dimensions[0], dimensions[1]))

    # use specified padding
    if pad_type == 'zero':
        # pad with zeros
        im = np.pad(im, padding, 'constant', constant_values=0)
    elif pad_type == 'repeat':
        # copy nextdoor pixel for padding
        im = np.pad(im, padding, 'symmetric')
    else:
        print('This should not have been reached')

    return im, im2  # return processed image


# post processes given image
# takes in image and whether or not to convert to rgb
def pp_image(im: np.ndarray, g2rgb: bool) -> np.ndarray:
    if g2rgb:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    # return pp image with correct type
    return im.astype(dtype='uint8')


# unused function
# designed to get neighbor pixels for given location and kernel size
# untested, may not work correctly
# takes in image, x and y location for desired pixel, and kernel size
# returns array containing window of specified kernel size centered on desired pixel
def get_pixel_neighbors(im: np.ndarray, x: int, y: int, kernel: int) -> np.ndarray:
    # create window array
    array = np.zeros((2 * kernel + 1, 2 * kernel + 1))
    # loop through window and get pixels
    for i in range(0, 2 * kernel + 1):
        for j in range(0, 2 * kernel + 1):
            array[i, j] = im[x - kernel + i, y - kernel + j]

    return array  # return neighboring pixels
