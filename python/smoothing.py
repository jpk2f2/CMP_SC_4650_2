import cv2
import matplotlib.pyplot as plot
import numpy as np


def prepare_image(im: np.ndarray, padded: bool) -> np.ndarray:
    # convert brg to grayscale
    if(len(im.shape)) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # zero pad unpadded image
    if not padded:
        im = np.pad(im, 1, 'constant', constant_values=0)
    # ensure image array is large enough to prevent overflow
    #im = im.astype(dtype='int64')
    return im.astype(dtype='int64')  # return processed image


def pp_image(im: np.ndarray, g2rgb: bool) -> np.ndarray:
    if g2rgb:
        im = im.cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    # return pp image with correct type
    return im.astype(dtype='uint8')


# 3x3 filter
# takes an image, true/false padding
# if more than one channel, strips it down to one
def box_filter(im: np.ndarray, padded: bool) -> np.ndarray:
    im = prepare_image(im, padded)  # prepare image
    # get dimensions of image
    dimensions = im.shape
    height, width = dimensions
    # loop through pixels(excluding zero pads)
    for i in range(1, height-1):
        for j in range(1, width-1):
            # add up all pixels in 3x3 area centered on current pixel
            top = im[i-1, j-1] + im[i-1, j] + im[i+1, j+1]
            mid = im[i, j-1] + im[i, j] + im[i, j+1]
            bot = im[i+1, j-1] + im[i+1, j] + im[i+1, j+1]
            total = top + mid + bot
            # average total
            avg = total/9
            # place rounded average back into the current pixel
            im[i, j] = round(avg)
    # postprocess image to correct type
    im = pp_image(im, False)
    return im  # return the processed image


def avg_filter(im: np.ndarray, padded: bool, ) -> np.ndarray:



    dimensions = im.shape
    height, width = dimensions
    for i in range(1, height-1):
        for j in range(1, width-1):
            top = im[i-1, j-1] * 0 + im[i-1, j]*1 + im[i+1, j+1]*0
            mid = im[i, j-1]*1 + im[i, j]*2 + im[i, j+1]*1
            bot = im[i+1, j-1]*0 + im[i+1, j]*1 + im[i+1, j+1]*0
            total = top + mid + bot
            avg = total/6
            im[i, j] = round(avg)
    return im.astype(dtype='uint8')