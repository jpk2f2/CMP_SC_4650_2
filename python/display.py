import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plot
import skimage.transform as transform
from typing import List  # additional typing support


# improved gaussian pyramid display function
# takes in a list containing the pyramid levels(images)
# processes, formats, and displays them
# can handle pyramids from 1 to 7 levels
def display_guass_pyramid(samples: List[np.ndarray]):
    len = samples.__len__()  # get number of levels
    # input check
    if len == 0:
        return

    # initial variables for formatting
    offset = 0
    padding = 1  # space between images in window
    gap = 64  # space between pyramid and top row
    square_size = 110  # size of squares in top row
    im_og_dim = samples[0].shape  # get width of largest(original) image
    dpi = 100  # dpi for display

    if len == 1:
        # only have to worry about single image
        width = im_og_dim[0]
    else:
        # ensure window is sized wide enough for display
        width = max(im_og_dim[0]+samples[1].shape[0], square_size*len)

    # get height of downsampled images for setting offset off bottom of window
    height = 0
    for sample in samples:
        height += sample.shape[1]

    offset = 2*im_og_dim[1] - height

    # set window width and height
    fig_w = width + padding*(len+1)
    fig_h = im_og_dim[1] + gap + 2*padding + square_size

    # create figure
    fig = plot.figure(figsize=(fig_w/dpi, fig_h/dpi))
    # place the original, largest image
    plot.figimage(samples[0], padding, padding, cmap='gray')

    # initalize x and y placement values
    xv = 0
    yv = im_og_dim[1] + 2  # ensure downsampled images line up with original image

    # format spacing and place pyramid images
    for i in range(1, len):
        xv = im_og_dim[0] + 2 * padding
        yv = yv - samples[i].shape[1] - 1
        plot.figimage(samples[i], xv, yv, cmap='gray')

    # setup and place top row of images
    # uses resize function from skimage to match image sizes
    # allows easy creation of top row to fully match example image
    for i in range(0, len):
        plot.figimage(transform.resize(samples[i], (square_size, square_size), anti_aliasing=False),
                      i * square_size + (i + 1) * padding, im_og_dim[1] + gap - padding, cmap='gray')

    plot.show()  # display figure
    return


# original, hardcoded gauss pyramid display logic
# was moved from main and slightly cleaned up
# use the current function display_gauss_pyramid() func instead
# takes in list containing pyrmaid levels
# processes, formats, and displays pyramid
def display_gauss_pyramid_defunct(sample: List[np.ndarray]):

    offset = 8
    padding = 1

    # fig = plot.figure(figsize=(7.68, 5.12))
    fig = plot.figure(figsize=(7.78, 6.86))
    plot.figimage(sample[0], 0 + padding, 0 + padding, cmap='gray')
    plot.figimage(sample[-1], 512 + 2 * padding, offset - 4 * padding, cmap='gray')
    plot.figimage(sample[-2], 512 + 2 * padding, 8 + offset - 3 * padding, cmap='gray')
    plot.figimage(sample[-3], 512 + 2 * padding, (8 + 16 + offset - 2 * padding), cmap='gray')
    plot.figimage(sample[-4], 512 + 2 * padding, (8 + 16 + 32 + offset - padding), cmap='gray')
    plot.figimage(sample[-5], 512 + 2 * padding, (8 + 16 + 32 + 64 + offset), cmap='gray')
    plot.figimage(sample[-6], 512 + 2 * padding, (8 + 16 + 32 + 64 + 128 + offset + padding), cmap='gray')

    plot.figimage(transform.resize(sample[0], (110, 110), anti_aliasing=False), 0 + padding, 512 + 64 - padding,
                  cmap='gray')
    plot.figimage(transform.resize(sample[1], (110, 110), anti_aliasing=False), 110 + 2 * padding, 512 + 64 - padding,
                  cmap='gray')
    plot.figimage(transform.resize(sample[2], (110, 110), anti_aliasing=False), 220 + 3 * padding, 512 + 64 - padding,
                  cmap='gray')
    plot.figimage(transform.resize(sample[3], (110, 110), anti_aliasing=False), 330 + 4 * padding, 512 + 64 - padding,
                  cmap='gray')
    plot.figimage(transform.resize(sample[4], (110, 110), anti_aliasing=False), 440 + 5 * padding, 512 + 64 - padding,
                  cmap='gray')
    plot.figimage(transform.resize(sample[5], (110, 110), anti_aliasing=False), 550 + 6 * padding, 512 + 64 - padding,
                  cmap='gray')
    plot.figimage(transform.resize(sample[6], (110, 110), anti_aliasing=False), 660 + 7 * padding, 512 + 64 - padding,
                  cmap='gray')

    plot.show()
    return
