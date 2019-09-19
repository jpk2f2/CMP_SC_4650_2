import cv2
import numpy as np


# preprocesses given image, returns padded version and array w/ original dimensions for final image
def prepare_image(im: np.ndarray, padding: int):
    dimensions = im.shape
    # convert brg to grayscale
    if (len(dimensions)) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # zero pad unpadded image
    im2 = np.zeros((dimensions[0], dimensions[1]))
    print(padding)
    im = np.pad(im, padding, 'constant', constant_values=0)
    # print(im)
    # print(im2.shape)
    # print(im.shape)
    # ensure image array is large enough to prevent overflow
    # im = im.astype(dtype='int64')
    return im.astype(dtype='int64'), im2  # return processed image


# post processes given image
def pp_image(im: np.ndarray, g2rgb: bool) -> np.ndarray:
    if g2rgb:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    # return pp image with correct type
    return im.astype(dtype='uint8')


# 3x3 filter
# takes an image, true/false padding
def box_filter(im: np.ndarray) -> np.ndarray:
    im_pad, final = prepare_image(im, 1)  # prepare image
    # get dimensions of image
    dimensions = im_pad.shape
    height, width = dimensions
    # loop through pixels(excluding zero pads)
    for i in range(0, height - 2):
        for j in range(0, width - 2):
            # add up all pixels in 3x3 area centered on current pixel
            top = im_pad[i - 1, j - 1] + im_pad[i - 1, j] + im_pad[i + 1, j + 1]
            mid = im_pad[i, j - 1] + im_pad[i, j] + im_pad[i, j + 1]
            bot = im_pad[i + 1, j - 1] + im_pad[i + 1, j] + im_pad[i + 1, j + 1]
            total = top + mid + bot
            # average total
            avg = total / 9
            # place rounded average back into the current pixel
            final[i, j] = round(avg)
    # postprocess image to correct type
    final = pp_image(final, False)
    return final  # return the processed image


# 3x3 filter
# takes image to be processed, true/false if image has been zero padded, and the designated mask in a 3x3 ndarray
# returns processed image
# a 3x3 array of all ones is equivalent to the box filter
def avg_filter(im: np.ndarray, mask: tuple) -> np.ndarray:
    array, padding = mask
    im, im2 = prepare_image(im, padding)  # preprocess image
    # get proper divisor by adding up weighted array values
    divisor = 0
    for i in array:
        for j in i:
            divisor = divisor + j

    # get image dimensions
    dimensions = im.shape
    height, width = dimensions
    # loops through image pixels, excluding zero edges
    for i in range(0+padding, height - 1 + padding):
        for j in range(0 + padding, width - 1 + padding):
            for k in range(0, padding):
                print(k)
            # add up each row w/ each entry multiplied by its designated weight
            top = im[i - 1, j - 1] * array[0, 0] + im[i - 1, j] * array[1, 0] + im[i + 1, j + 1] * array[2, 0]
            mid = im[i, j - 1] * array[0, 1] + im[i, j] * array[1, 1] + im[i, j + 1] * array[2, 1]
            bot = im[i + 1, j - 1] * array[0, 2] + im[i + 1, j] * array[1, 2] + im[i + 1, j + 1] * array[2, 2]
            total = top + mid + bot  # add up rows
            avg = total / divisor  # average total using calculated divisor
            im2[i, j] = round(avg)  # round result
    # postprocess image to fix type
    im2 = pp_image(im2, False)
    return im2  # return processed image


# def guass_filter(im: ndarray, padded: bool, m)