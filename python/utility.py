import cv2
import numpy as np
import math


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

    return im.astype(dtype='float64'), im2.astype(dtype='float64')  # return processed image


# post processes given image
# takes in image and whether or not to convert to rgb
def pp_image(im: np.ndarray, g2rgb: bool) -> np.ndarray:
    if g2rgb:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    # return pp image with correct type
    return im.astype(dtype='uint8')


# creates a gaussian matrix using a given kernel size for convulation w/ an image
# returns the guassian matrix
# for example, a kernel size of 2 creates a 5x5 gaussian array
def create_gauss_conv(kernel: int) -> np.ndarray:
    # create array to hold guassian array
    final = np.zeros(shape=(2 * kernel + 1, 2 * kernel + 1))
    tot = 0
    # loop through array, creating gaussian distribution
    for i in range(-kernel, kernel + 1):
        for j in range(-kernel, kernel + 1):
            # create guassian dividend
            tmp = -1 * ((i ** 2 + j ** 2) / (2 * (kernel ** 2)))
            # complete gaussian function and place it in dest
            final[i + kernel, j + kernel] = math.exp(tmp) / (2 * np.pi * kernel ** 2)
            # count total for normalization
            tot = tot + final[i + kernel, j + kernel]

    # normalize gaussian array
    final = final / tot

    return final


# unused function
# designed to get neighbor pixels for given location and kernel size
# untested and never used, may not work correctly
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
