# Jason Kayser
# jpk2f2
import cv2
import numpy as np
import matplotlib.pyplot as plot

# import own libraries
# import display as _display
import smoothing as _smooth
import masks

# read in part 1 images
fig_312 = cv2.imread('resources/Fig0312(a)(kidney).tif', 0)
fig_333 = cv2.imread('resources/Fig0333(a)(test_pattern_blurring_orig).tif', 0)
fig_334 = cv2.imread('resources/Fig0334(a)(hubble-original).tif')
fig_335 = cv2.imread('resources/Fig0335(a)(ckt_board_saltpep_prob_pt05).tif')
#read in part 2 images
fig_338 = cv2.imread('resources/Fig0338(a)(blurry_moon).tif')
fig_340 = cv2.imread('resources/Fig0340(a)(dipxe_test).tif')

im_prop = fig_312.shape
#height, width, channels = im_prop
height, width = im_prop
# flat_im = fig_312.flatten()

# print(im_prop)
# print(fig_312)
# print(flat_im)

fig_312_pad = np.pad(fig_312, 1, 'constant', constant_values=0)
fig_333_pad = np.pad(fig_333, 1, 'constant', constant_values=0)

blurred_fig_333 = _smooth.box_filter(fig_333_pad.astype(dtype='int64'), 0)
cv2.imshow('before', fig_333)
cv2.imshow('after', blurred_fig_333)
cv2.waitKey(0)
