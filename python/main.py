# Jason Kayser
# jpk2f2
import cv2
import matplotlib
# import own libraries
import display
import pyramid

# use TkAgg backend to display in window instead of pycharm scientific mode
matplotlib.use('TkAgg')

# read in part 1 images
fig_312 = cv2.imread('resources/Fig0312(a)(kidney).tif', 0)
fig_333 = cv2.imread('resources/Fig0333(a)(test_pattern_blurring_orig).tif', 0)
fig_334 = cv2.imread('resources/Fig0334(a)(hubble-original).tif', 0)
fig_335 = cv2.imread('resources/Fig0335(a)(ckt_board_saltpep_prob_pt05).tif', 0)
# read in part 2 images
fig_338 = cv2.imread('resources/Fig0338(a)(blurry_moon).tif', 0)
fig_340 = cv2.imread('resources/Fig0340(a)(dipxe_text).tif', 0)
# read in part 3 images
lenna = cv2.imread('resources/Lenna.png', 0)

# displays for inclass presenting
# display median filter at different kernel sizes for all four images
# note that this runs quite slow, especially in comparison to the rest
display.display_med_filter([fig_312, fig_333, fig_334, fig_335])
# display average(box) filter at different kernel sizes for all four images
display.display_avg_filter([fig_312, fig_333, fig_334, fig_335])
# display average(weighted) filter at different kernel sizes for all four images
display.display_avg_filter([fig_312, fig_333, fig_334, fig_335], True)
# display gaussian filter at different kernel sizes for all four images
display.display_gauss_filter([fig_312, fig_333, fig_334, fig_335])
# display sharpening filter for both images
display.display_lpl_filter([fig_338, fig_340])
# display gaussian pyramid for 'lenna' image
tmp = pyramid.get_gaussian_pyramid(lenna, 7)
display.display_guass_pyramid(tmp)
