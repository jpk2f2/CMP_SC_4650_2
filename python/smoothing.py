import cv2
import numpy as np
import math
import scipy.signal
import masks
from utility import prepare_image, pp_image


# 3x3 box filter
# takes an image
# returns processed image
def box_filter(im: np.ndarray) -> np.ndarray:
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


# 3x3 filter
# takes image to be processed, true/false if image has been zero padded, and the designated mask in a 3x3 ndarray
# returns processed image
# a 3x3 array of all ones is equivalent to the box filter
def avg_filter(im: np.ndarray, mask) -> np.ndarray:
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
    ex_2 = scipy.signal.convolve2d(im, _filter)
    ex_2 = pp_image(ex_2, False)
    im, im2 = prepare_image(im, padding, 'zero')  # preprocess image

    dimensions = im.shape
    height, width = dimensions
    ex = scipy.signal.convolve2d(im, _filter, 'valid')
    ex = pp_image(ex, False)
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
    im = pp_image(im, False)
    cv2.imshow('orig?: ', im)
    cv2.imshow('ex: ', ex)
    cv2.imshow('own: ', im2)
    cv2.imshow('ex2: ', ex_2)
    cv2.waitKey(0)
    return im2  # return processed image


# median filter
# takes in the image and kernel size
# kernel size equates to a 2*kernel+1 by 2*kernel+1 sliding window ie kernel 1 = 3x3 window
# handles any kernel size
# returns processed image
def med_filter(im: np.ndarray, kernel: int) -> np.ndarray:
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
