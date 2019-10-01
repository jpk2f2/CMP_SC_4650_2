import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import skimage.transform as transform
from skimage import exposure
from typing import List  # additional typing support

import smoothing as smooth
import masks
import sharpening as sharpen


matplotlib.use('TkAgg')  # print figures to window instead of pycharm scientific mode

"""
Display functions for all the first two parts of the assignment
To be filled in later

def display_avg_filter(im: np.ndarray, weighted: bool):

def display_med_filter(im: np.ndarray):

def display_gauss_filter(im: np.ndarray, premade: bool):

def display_lpl_sharpen(im: np.ndarray):

# this algorithm does not exist yet
# may never be used
def display_unsharp(im: np.ndarray):


"""


def display_med_filter(ims: List[np.ndarray]):
    ctr = 1
    for im in ims:
        print("image {} is being processed".format(ctr))
        f, axe = plot.subplots(1, 4, squeeze=True)
        f.set_tight_layout(True)
        f.suptitle('median filter', fontsize=16)
        axe[0].imshow(im, cmap='gray')
        axe[0].set_title('Before: ')
        axe[0].axis('off')
        for i in range(1, 4):
            axe[i].imshow(smooth.med_filter(im, i), cmap='gray')
            axe[i].set_title('After kernel {}: '.format(i))
            axe[i].axis('off')
            axe[i].set_aspect(aspect=1)

        manager = plot.get_current_fig_manager()
        manager.full_screen_toggle()
        print("image {} has been processed".format(ctr))
        ctr += 1
    plot.show()
    return


def display_avg_filter(ims: List[np.ndarray], weighted: bool = False):
    for im in ims:
        f, axe = plot.subplots(1, 2, squeeze=True) if weighted else plot.subplots(1, 4, squeeze=True)
        f.set_tight_layout(True)
        f.suptitle('average filter - weighted:{}'.format(weighted), fontsize=16)
        axe[0].imshow(im, cmap='gray')
        axe[0].set_title('Before: ')
        axe[0].axis('off')
        for i in range(1, 2 if weighted else 4):
            axe[i].imshow(smooth.avg_filter(im, i, weighted), cmap='gray')
            axe[i].set_title('After kernel {}: '.format(i))
            axe[i].axis('off')
            axe[i].set_aspect(aspect=1)

            # cv2.imshow('Before: ', im)
            # cv2.imshow('After kernel {}: '.format(i), smooth.avg_filter(im, i, weighted))
        manager = plot.get_current_fig_manager()
        manager.full_screen_toggle()
        # manager.resize(*manager.window.maxsize())
        plot.show()
    return


def display_gauss_filter(ims: List[np.ndarray]):
    for im in ims:
        f, axe = plot.subplots(1, 4, squeeze=True)
        f.set_tight_layout(True)
        f.suptitle('gaussian filter', fontsize=16)
        axe[0].imshow(im, cmap='gray')
        axe[0].set_title('Before: ')
        axe[0].axis('off')
        for i in range(1, 4):
            axe[i].imshow(smooth.guass_filter(im, i), cmap='gray')
            axe[i].set_title('After kernel {}: '.format(i))
            axe[i].axis('off')
            axe[i].set_aspect(aspect=1)

        manager = plot.get_current_fig_manager()
        manager.full_screen_toggle()
        # manager.resize(*manager.window.maxsize())
        plot.show()
    return


def display_lpl_filter(ims: List[np.ndarray]):
    for im in ims:
        f, axe = plot.subplots(1, 4, squeeze=True)
        f.set_tight_layout(True)
        f.suptitle('laplacian sharpening', fontsize=16)
        axe[0].imshow(im, cmap='gray')
        axe[0].set_title('Before: ')
        axe[0].axis('off')
        axe[0].set_aspect(aspect=1)

        final, lpl = sharpen.lpl_sharpen(im, masks.LPL2_3X3)
        axe[1].imshow(lpl, cmap='gray')
        axe[1].set_title('laplacian filter')
        axe[1].axis('off')
        axe[1].set_aspect(aspect=1)

        axe[2].imshow(final, cmap='gray')
        axe[2].set_title('sharpened image')
        axe[2].axis('off')
        axe[2].set_aspect(aspect=1)

        p2, p98 = np.percentile(final, (1, 99))
        img_rescale = exposure.rescale_intensity(final, in_range=(p2, p98))

        axe[3].imshow(img_rescale, cmap='gray')
        axe[3].set_title('after contrast stretch')
        axe[3].axis('off')
        axe[3].set_aspect(aspect=1)

        manager = plot.get_current_fig_manager()
        manager.full_screen_toggle()
        plot.show()
    return


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
    # make figure fullscreen
    manager = plot.get_current_fig_manager()
    manager.full_screen_toggle()
    plot.suptitle('pyramid', fontsize=16)
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
