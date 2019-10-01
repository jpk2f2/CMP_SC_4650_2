import numpy as np

# additional typing support
from typing import List

# my modules
import utility
import smoothing


# downsample image for gaussian pyramid
# takes in image and returns downsampled image
# downsamples by running image through a gaussian smoothing algorithm of kernel size 1
# and then reads every other pixel of blurred image for return
def guass_pyramid_resize(im: np.ndarray) -> np.ndarray:
    im = smoothing.guass_filter(im, 1, False)  # smooth image with kernel 1 gaussian
    # get halved dimensions for downsampled image
    dimensions = im.shape
    im_ds = np.zeros((dimensions[0]//2, dimensions[1]//2))
    dimensions = im_ds.shape

    # get every other pixel from blurred image
    for i in range(0, dimensions[0]):
        for j in range(0, dimensions[1]):
            im_ds[i, j] = im[2 * i, 2 * j]

    im_ds = utility.pp_image(im_ds, False)  # run post-processing

    return im_ds  # return downsampled image


# create gaussian pyramid with specified number of levels
# takes in an image and the number of levels desired for the pyramid
# returns a list containing all levels of pyramid in order of decreasing resolution
# max 7 levels with the first being the original image
def get_gaussian_pyramid(im: np.ndarray, levels: int) -> List[np.ndarray]:
    # input checking
    if levels > 7:
        levels = 7
    if levels <= 1:  # hit end of recursive or no levels
        return [im, ]

    # recusively call function
    x = get_gaussian_pyramid(im, levels-1)
    # append processed image
    x.append(guass_pyramid_resize(x[-1]))
    return x

