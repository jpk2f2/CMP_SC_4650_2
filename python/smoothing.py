import numpy as np
import math
import scipy.signal
import masks
from multiprocessing import Process, Manager
from utility import prepare_image, pp_image, create_gauss_conv


# average smoothing algorithm
# takes in an image, an optional tuple containing a mask and it's appropriate kernel, and an optional tuple with a
# bool stating whether a box filter should be used alongside the kernel size if True
# if no mask is specified and box filter isn't specified it defaults to a 3x3 box filter
# available mask/kernel tuples can be found in masks.py, or custom can be used following in their pattern
# returns processed image
def avg_filter(im: np.ndarray, kernel: int = 1, weighted: bool = False, w_alt: bool = True):
    if weighted:
        if kernel != 1:
            return
        _filter = masks.WTD1_3X3 if w_alt else masks.WTD2_3X3
    else:
        _filter = np.ones((2*kernel+1, 2*kernel+1), dtype='int')

    im, im_proc = prepare_image(im, kernel, 'zero')  # preprocess image
    # loop through filter to get divisor
    div = 0  # init
    for i in _filter:
        for j in i:
            div = div + j

    im_proc = scipy.signal.convolve2d(im, _filter, 'valid')
    im_proc /= div

    return pp_image(im_proc, False)


# median filter
# currently mostly identical to the defunct version
# working on ways to make it more efficient, i.e not O(n^4)...yikes
# takes in the image and kernel size
# kernel size equates to a 2*kernel+1 by 2*kernel+1 sliding window ie kernel 1 = 3x3 window
# handles any kernel size
# returns processed image
def med_filter(im: np.ndarray, kernel: int) -> np.ndarray:
    im, im_proc = prepare_image(im, kernel, 'zero')  # preprocess image
    dimensions = im.shape
    # loop through image pixels, excluding padded edges
    # place pixels within sliding window in list
    window = []
    for i in range(0 + kernel, dimensions[0] - (2 * kernel)):
        for j in range(0 + kernel, dimensions[1] - (2 * kernel)):
            for k in range(-kernel, kernel + 1):
                for l in range(-kernel, kernel + 1):
                    window.append(im[i + k, j + l])
            # sort pixels from sliding window and then place median value in processed image
            window.sort()
            im_proc[i, j] = window[math.floor(((kernel * 2 + 1) ** 2) / 2)]
            window.clear()  # clear sliding window list for next iteration
    # return processed image
    return pp_image(im_proc, False)


"""
def med_filter_2(im: np.ndarray, kernel: int) -> np.ndarray:
    im, im_proc = prepare_image(im, kernel, 'zero')  # preprocess image
    dimensions = im.shape
    # loop through image pixels, excluding padded edges
    # place pixels within sliding window in list
    window = []
    # b = np.zeros()
    for i in range(0 + kernel, dimensions[0] - (2 * kernel)):
        for j in range(0 + kernel, dimensions[1] - (2 * kernel)):
            a = []
            for k in range(-kernel, kernel + 1):
                for l in range(-kernel, kernel + 1):
                    # a[k+kernel, l+kernel] = [i+k, j+l]
                    a.append([i+k, j+l])
            print(kernel)
            print(a.__len__())
            print(a)
            print(np.take(im, a))
            #im_proc[i, j] = np.median(b)
            # sort pixels from sliding window and then place median value in processed image
    # return processed image
    return pp_image(im_proc, False)
"""


# generic guassian filter algorithm
# takes in image, kernel size, and whether to use a premade filter( max premade size of 3)
# kernel size equates to a size*2+1 by size*2+1 sliding window
# returns process image
def guass_filter(im: np.ndarray, kernel: int, premade: bool = False) -> np.ndarray:
    # check if premade desired
    if premade:
        _filter = masks.kernel_to_filter(kernel)  # use premade guassian filter
    else:
        _filter = create_gauss_conv(kernel)  # create a guassian filter for the given kernel

    im, im_proc = prepare_image(im, kernel, 'zero')  # preprocess image
    # convolve image with filter
    # 'valid' specified to use preprocessed padding
    im_proc = scipy.signal.convolve2d(im, _filter, 'valid')

    # post process image and return it
    return pp_image(im_proc, False)


"""
    All algorithms below this line are no longer being actively used
    They have been kept for either testing, for the professor to view, or for posterity
    Reccomended to use the more flexible and efficient versions above
"""


# guassian filter
# takes in the image and kernel size
# kernel size equates to a 2*kernel+1 by 2*kernel+1 sliding window ie kernel 1 = 3x3 window
# handles kernel size 1 only
# returns processed image
def guass_filter_1(im: np.ndarray, kernel: int) -> np.ndarray:
    # hardcode kernel size just in case
    kernel = 1
    # use premade gaussian filter for kernel 1
    _filter = masks.GAUS_3X3
    # create an appropriate guassian filter for given kernel size
    # _filter = create_gauss_conv(kernel)
    padding = kernel
    im, im2 = prepare_image(im, padding, 'zero')  # preprocess image

    dimensions = im.shape
    height, width = dimensions
    # loops through image pixels, excluding zero edges
    for i in range(0 + padding, height - (2 * padding)):
        for j in range(0 + padding, width - (2 * padding)):
            # hardcoded calcs and dimensions
            top = im[i - 1, j - 1] * _filter[0, 0] + im[i - 1, j] * _filter[1, 0] + im[i - 1, j + 1] * _filter[2, 0]
            mid = im[i, j - 1] * _filter[0, 1] + im[i, j] * _filter[1, 1] + im[i, j + 1] * _filter[2, 1]
            bot = im[i + 1, j - 1] * _filter[0, 2] + im[i + 1, j] * _filter[1, 2] + im[i + 1, j + 1] * _filter[2, 2]
            total = top + mid + bot  # add up rows
            im2[i, j] = round(total)  # round result
    # postprocess image to fix type
    im2 = pp_image(im2, False)
    return im2  # return processed image


# guassian filter
# takes in the image and kernel size
# kernel size equates to a 2*kernel+1 by 2*kernel+1 sliding window ie kernel 1 = 3x3 window
# handles kernel size 3 only
# returns processed image
def guass_filter_3(im: np.ndarray, kernel: int) -> np.ndarray:
    # hardcode kernel just in case
    kernel = 3
    # use premade gaussian filter
    # it's already normalized
    _filter = masks.GAUS_7X7
    # calculate gaussian filter for specified kernel
    # _filter = create_gauss_conv(kernel)
    padding = kernel
    im, im2 = prepare_image(im, padding, 'zero')  # preprocess image

    dimensions = im.shape
    height, width = dimensions
    # loops through image pixels, excluding zero edges
    for i in range(0 + padding, height - (2 * padding)):
        for j in range(0 + padding, width - (2 * padding)):
            # harcoded calculations and dimensions
            line1 = im[i - 3, j - 3] * _filter[0, 0] + im[i - 3, j - 2] * _filter[1, 0] + \
                    im[i - 3, j - 1] * _filter[2, 0] + im[i - 3, j] * _filter[3, 0] + \
                    im[i - 3, j + 1] * _filter[4, 0] + im[i - 3, j + 2] * _filter[5, 0] + \
                    im[i - 3, j + 3] * _filter[6, 0]
            line2 = im[i - 2, j - 3] * _filter[0, 1] + im[i - 2, j - 2] * _filter[1, 1] + \
                    im[i - 2, j - 1] * _filter[2, 1] + im[i - 2, j] * _filter[3, 1] + \
                    im[i - 2, j + 1] * _filter[4, 1] + im[i - 2, j + 2] * _filter[5, 1] + \
                    im[i - 2, j + 3] * _filter[6, 1]
            line3 = im[i - 1, j - 3] * _filter[0, 2] + im[i - 1, j - 2] * _filter[1, 2] + \
                    im[i - 1, j - 1] * _filter[2, 2] + im[i - 1, j] * _filter[3, 2] + \
                    im[i - 1, j + 1] * _filter[4, 2] + im[i - 1, j + 2] * _filter[5, 2] + \
                    im[i - 1, j + 3] * _filter[6, 2]
            line4 = im[i, j - 3] * _filter[0, 3] + im[i, j - 2] * _filter[1, 3] + \
                    im[i, j - 1] * _filter[2, 3] + im[i, j] * _filter[3, 3] + \
                    im[i, j + 1] * _filter[4, 3] + im[i, j + 2] * _filter[5, 3] + \
                    im[i, j + 3] * _filter[6, 3]
            line5 = im[i + 1, j - 3] * _filter[0, 4] + im[i + 1, j - 2] * _filter[1, 4] + \
                    im[i + 1, j - 1] * _filter[2, 4] + im[i + 1, j] * _filter[3, 4] + \
                    im[i + 1, j + 1] * _filter[4, 4] + im[i + 1, j + 2] * _filter[5, 4] + \
                    im[i + 1, j + 3] * _filter[6, 4]
            line6 = im[i + 2, j - 3] * _filter[0, 5] + im[i + 2, j - 2] * _filter[1, 5] + \
                    im[i + 2, j - 1] * _filter[2, 5] + im[i + 2, j] * _filter[3, 5] + \
                    im[i + 2, j + 1] * _filter[4, 5] + im[i + 2, j + 2] * _filter[5, 5] + \
                    im[i + 2, j + 3] * _filter[6, 5]
            line7 = im[i + 3, j - 3] * _filter[0, 6] + im[i + 3, j - 2] * _filter[1, 6] + \
                    im[i + 3, j - 1] * _filter[2, 6] + im[i + 3, j] * _filter[3, 6] + \
                    im[i + 3, j + 1] * _filter[4, 6] + im[i + 3, j + 2] * _filter[5, 6] + \
                    im[i + 3, j + 3] * _filter[6, 6]
            total = line1 + line2 + line3 + line4 + line5 + line6 + line7  # add up rows
            im2[i, j] = round(total)  # round result
    # postprocess image to fix type
    im2 = pp_image(im2, False)
    # im = pp_image(im, False)
    return im2  # return processed image


# median filter
# takes in the image and kernel size
# kernel size equates to a 2*kernel+1 by 2*kernel+1 sliding window ie kernel 1 = 3x3 window
# handles any kernel size
# returns processed image
def med_filter_defunct(im: np.ndarray, kernel: int) -> np.ndarray:
    padding = kernel
    im, im2 = prepare_image(im, padding, 'zero')  # preprocess image
    # get image dimensions
    dimensions = im.shape
    height, width = dimensions
    # loops through image pixels, excluding zero edges
    window = []  # intialize list var
    for i in range(0 + padding, height - (2 * padding)):
        for j in range(0 + padding, width - (2 * padding)):
            # loop through sliding window size to handle various kernel sizes
            for k in range(-kernel, kernel + 1):
                for l in range(-kernel, kernel + 1):
                    window.append(im[i + k, j + l])  # add current pixel to list
            window.sort()  # sort list for median selection
            # calculate median location and it in dest pixel
            im2[i, j] = window[math.floor(((kernel * 2 + 1) ** 2) / 2)]
            window.clear()  # clear list for next calc
    # postprocess image to fix type
    im2 = pp_image(im2, False)
    return im2  # return processed image


# 3x3 filter average filter
# takes image to be processed, true/false if image has been zero padded, and the designated mask in a 3x3 ndarray
# returns processed image
# a 3x3 array of all ones is equivalent to the box filter

def avg_filter_defunct(im: np.ndarray, mask) -> np.ndarray:
    array = mask[0]
    # padding = 1
    padding = mask[1]
    # array, padding = mask
    im, im2 = prepare_image(im, padding, 'zero')  # preprocess image
    # get proper divisor by adding up weighted array values
    divisor = 0
    for i in array:
        for j in i:
            divisor = divisor + j

    # get image dimensions
    dimensions = im.shape
    height, width = dimensions
    # loops through image pixels, excluding zero edges
    for i in range(0 + padding, height - (1 + padding)):
        for j in range(0 + padding, width - (1 + padding)):
            # add up each row w/ each entry multiplied by its designated weight
            # hardcoded
            top = im[i - 1, j - 1] * array[0, 0] + im[i - 1, j] * array[1, 0] + im[i + 1, j + 1] * array[2, 0]
            mid = im[i, j - 1] * array[0, 1] + im[i, j] * array[1, 1] + im[i, j + 1] * array[2, 1]
            bot = im[i + 1, j - 1] * array[0, 2] + im[i + 1, j] * array[1, 2] + im[i + 1, j + 1] * array[2, 2]
            total = top + mid + bot  # add up rows
            avg = total / divisor  # average total using calculated divisor
            im2[i, j] = round(avg)  # round result
    # postprocess image to fix type
    im2 = pp_image(im2, False)
    return im2  # return processed image


# 3x3 box filter
# use average filter instead and specify box filter
# takes an image
# returns processed image
def box_filter_defunct(im: np.ndarray) -> np.ndarray:
    im_pad, final = prepare_image(im, 1, 'zero')  # prepare image
    # get dimensions of image
    dimensions = im_pad.shape
    height, width = dimensions
    # loop through pixels(excluding zero pads)
    for i in range(0, height - 2):
        for j in range(0, width - 2):
            # add up all pixels in 3x3 area centered on current pixel
            # hardcoded
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
