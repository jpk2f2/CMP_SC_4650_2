# Jason Kayser
# jpk2f2
import cv2
import numpy as np
import matplotlib.pyplot as plot

# import own libraries
import display as _display
import smoothing as _smooth

# read in part 1 images
fig_312 = cv2.imread('resources/Fig0312(a)(kidney).tif')
fig_333 = cv2.imread('resources/Fig0333(a)(test_pattern_blurring_orig).tif')
fig_334 = cv2.imread('resources/Fig0334(a)(hubble-original).tif')
fig_335 = cv2.imread('resources/Fig0335(a)(ckt_board_saltpep_prob_pt05).tif')
#read in part 2 images
fig_338 = cv2.imread('resources/Fig0338(a)(blurry_moon).tif')
fig_340 = cv2.imread('resources/Fig0340(a)(dipxe_test).tif')

