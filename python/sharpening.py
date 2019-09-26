import masks
import numpy as np
import cv2
import math

from utility import prepare_image, pp_image


def lpl_sharpen_1(im: np.ndarray, mask) -> np.ndarray:
    array = mask[0]
    # padding = 1
    padding = mask[1]
    # array, padding = mask
    im2 = im.copy()
    im2 = im2.astype('int64')
    im, im4 = prepare_image(im, padding)  # preprocess image
    # get proper divisor by adding up weighted array values
    dimensions = im2.shape
    im3 = np.zeros((dimensions[0], dimensions[1]))
    # im2 = im2.astype('float64')
    # get image dimensions
    dimensions = im.shape
    height, width = dimensions
    # loops through image pixels, excluding zero edges
    for i in range(0 + padding, height - (1 + padding)):
        for j in range(0 + padding, width - (1 + padding)):
            # add up each row w/ each entry multiplied by its designated weight
            # hardcoded
            top = im[i - 1, j - 1] * array[0, 0] + im[i - 1, j] * array[1, 0] + im[i - 1, j + 1] * array[2, 0]
            mid = im[i, j - 1] * array[0, 1] + im[i, j] * array[1, 1] + im[i, j + 1] * array[2, 1]
            bot = im[i + 1, j - 1] * array[0, 2] + im[i + 1, j] * array[1, 2] + im[i + 1, j + 1] * array[2, 2]
            total = top + mid + bot  # add up rows
            # int("total: {}".format(total))
            im3[i, j] = total
    cv2.imshow("test1", im3)
    cv2.waitKey(0)
    # print("min: {}".format(im2.min()))
    im3 -= np.min(im3)
    # print("min: {}".format(im2.min()))
    cv2.imshow("test2", im3)
    cv2.waitKey(0)
    # print("max: {}".format(im2.max()))
    im3 /= im3.max()
    # print("max: {}".format(im2.max()))
    im3 *= 255.0

    # im3 = im3.astype('uint8')
    im2 = im2 + im3
    im2 -= np.min(im2)
    im2 /= im2.max()
    im2 *= 255.0
    im2 = im2.astype('uint8')
    cv2.imshow("test3", im2)
    cv2.waitKey(0)
    # postprocess image to fix type
    im2 = pp_image(im2, False)
    return im2  # return processed image