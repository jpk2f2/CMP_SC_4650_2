import cv2
import numpy as np

import pyramid
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plot
import skimage.transform as transform
from typing import List


def display_guass_pyramid(samples: List[np.ndarray]):
    len = samples.__len__()
    if len == 0:
        return
    offset = 0
    padding = 1
    gap = 64
    square_size = 110
    im_og_dim = samples[0].shape
    dpi = 100

    if len == 1:
        width = im_og_dim[0]
    else:
        width = max(im_og_dim[0]+samples[1].shape[0], square_size*len)
    height = 0
    for sample in samples:
        height += sample.shape[1]

    offset = 2*im_og_dim[1] - height
    fig_w = width + padding*(len+1)
    fig_h = im_og_dim[1] + gap + 2*padding + square_size

    fig = plot.figure(figsize=(fig_w/dpi, fig_h/dpi))
    # place the original, largest image
    plot.figimage(samples[0], padding, padding, cmap='gray')

    xv = 0
    yv = im_og_dim[1] + 2
    for i in range(1, len):
        xv = im_og_dim[0] + 2 * padding
        yv = yv - samples[i].shape[1] - 1
        plot.figimage(samples[i], xv, yv, cmap='gray')

    for i in range(0, len):
        plot.figimage(transform.resize(samples[i], (square_size, square_size), anti_aliasing=False),
                      i * square_size + (i + 1) * padding, im_og_dim[1] + gap - padding, cmap='gray')

    plot.show()


def display_guass_pyramid_defunct(sample: List[np.ndarray]):

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