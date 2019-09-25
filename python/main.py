# Jason Kayser
# jpk2f2
import cv2
import numpy as np

# import own libraries
# import display as display
import smoothing as smooth
import sharpening as sharpen
import masks  # masks/kernels for use with filtering algorithms

# read in part 1 images
fig_312 = cv2.imread('resources/Fig0312(a)(kidney).tif', 0)
fig_333 = cv2.imread('resources/Fig0333(a)(test_pattern_blurring_orig).tif', 0)
fig_334 = cv2.imread('resources/Fig0334(a)(hubble-original).tif', 0)
fig_335 = cv2.imread('resources/Fig0335(a)(ckt_board_saltpep_prob_pt05).tif', 0)
# read in part 2 images
fig_338 = cv2.imread('resources/Fig0338(a)(blurry_moon).tif', 0)
fig_340 = cv2.imread('resources/Fig0340(a)(dipxe_test).tif', 0)

im_prop = fig_312.shape
# height, width, channels = im_prop
height, width = im_prop
# flat_im = fig_312.flatten()

# print(im_prop)
# print(fig_312)
# print(flat_im)

# fig_312_pad = np.pad(fig_312, 1, 'constant', constant_values=0)
# fig_333_pad = np.pad(fig_333, 1, 'constant', constant_values=0)

# blurred_fig_333 = smooth.box_filter(fig_333)
# blurred_fig_3332 = smooth.avg_filter(fig_333, masks.WTD1_3X3)
# blurred_fig_3333 = smooth.guass_filter(fig_333, 1)
# blurred_fig_3334 = smooth.guass_filter_3(fig_333, 3)
blurred_fig_3335 = smooth.med_filter(fig_335, 1)
sharpened_fig_338 = sharpen.lpl_sharpen_1(fig_338, masks.LPL2_3X3)
# cv2.imshow('before', fig_333)
# cv2.imshow('after box', blurred_fig_333)
# cv2.imshow('after weighted avg', blurred_fig_3332)
# cv2.imshow('after gauss', blurred_fig_3333)
# cv2.imshow('after gauss_3', blurred_fig_3334)
cv2.imshow('after median', blurred_fig_3335)
cv2.imshow('before sharpen', fig_338)
cv2.imshow('after sharpen', sharpened_fig_338)
cv2.waitKey(0)

#  print(smooth.create_gauss_conv(3))
