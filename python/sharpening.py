import numpy as np
import scipy.signal

from utility import prepare_image, pp_image


# runs a laplacian sharpen algorithm on a given image using a given filter
# takes in an image and a tuple containing an ndarray filter and the kernel/padding size
# there is a list of premade filters in the masks.py module
# submit your own filter with its kernel size in the format (ndarray,int)
# returns the final sharpened image and the normalized laplacian filter result
def lpl_sharpen(im: np.ndarray, mask) -> (np.ndarray, np.ndarray):
    array, padding = mask  # get filter, kernel/mask
    im_final = im.copy()  # copy image for use in sharpened image before padding
    im, im_lpl = prepare_image(im, padding, 'repeat')  # preprocess images

    # code below is my logic for convulation
    # it is commented out so that the more efficient scipy convolve2d function can be used
    # code below being saved for future improvement and/or comparison with convolve2d
    """
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
            im3[i, j] = total

    """
    # convolve image with filter
    # specify 'valid' to use prepared padding and prevent double padding
    im_lpl = scipy.signal.convolve2d(im, array, 'valid')
    im_lpl = im_lpl.astype('float64')  # fix type for upcoming manipulation

    # display original laplacian sharpen outcome
    # cv2.imshow('test: ', im_lpl.astype('uint8'))
    # cv2.imshow('test2: ', im_lpl.astype('byte'))
    # cv2.waitKey(0)

    # normalize values
    im_lpl -= np.min(im_lpl)
    im_lpl /= np.max(im_lpl)
    im_lpl *= 255.0

    # display normalized laplacian sharpen outcome
    # cv2.imshow('test: ', im_lpl.astype('uint8'))
    # cv2.imshow('test2: ', im_lpl.astype('byte'))
    # cv2.waitKey(0)

    # add laplacian sharpen to image for final result
    im_final = im_final + im_lpl
    # normalize sharpened image
    im_final -= np.min(im_final)
    im_final /= np.max(im_final)
    im_final *= 255.0

    # postprocess images to fix type
    im_final = pp_image(im_final, False)
    im_lpl = pp_image(im_lpl, False)

    return im_final, im_lpl  # return processed images
