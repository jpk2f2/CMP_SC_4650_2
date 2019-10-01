# Jason Kayser
# jpk2f2
import cv2
import matplotlib
# import own libraries
import display
import pyramid

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


display.display_med_filter([fig_312, fig_333, fig_334, fig_335])
display.display_avg_filter([fig_312, fig_333, fig_334, fig_335])
display.display_avg_filter([fig_312, fig_333, fig_334, fig_335], True)
display.display_gauss_filter([fig_312, fig_333, fig_334, fig_335])
display.display_lpl_filter([fig_338, fig_340])
tmp = pyramid.get_gaussian_pyramid(lenna, 7)
display.display_guass_pyramid(tmp)


#  print(smooth.create_gauss_conv(3))
