import numpy as np

STD_3X3 = np.ones((3, 3), dtype='int'), 1
STD_5X5 = np.ones((5, 5), dtype='int'), 2
STD_7X7 = np.ones((7, 7), dtype='int'), 3

# weighted standard filter
WTD1_3X3 = [0, 1, 0, 1, 2, 1, 0, 1, 0]
WTD1_3X3 = np.array(WTD1_3X3).reshape((3, 3))
WTD1_3X3 = (WTD1_3X3, 1)
# WTD1_3X3_2 = WTD1_3X3, 1

# weighted standard filter ver. 2
WTD2_3X3 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
WTD2_3X3 = np.array(WTD2_3X3).reshape((3, 3))
WTD2_3X3 = (WTD2_3X3, 1)

# Laplacian filters with their kernel sizes

# Laplacian filter 1 and its alt
LPL1_3X3 = [0, -1, 0, -1, 4, -1, 0, -1, 0]
LPL1_3X3 = np.array(LPL1_3X3).reshape((3, 3))
LPL1_3X3 = (LPL1_3X3, 1)
LPL1_3X3_ALT = [0, 1, 0, 1, -4, 1, 0, 1, 0]
LPL1_3X3_ALT = np.array(LPL1_3X3_ALT).reshape((3, 3))
LPL1_3X3_ALT = (LPL1_3X3_ALT, 1)

# Laplacian filter 2 and its alt
LPL2_3X3 = [-1, -1, -1, -1, 8, -1, -1, -1, -1]
LPL2_3X3 = np.array(LPL2_3X3).reshape((3, 3))
LPL2_3X3 = (LPL2_3X3, 1)
LPL2_3X3_ALT = [1, 1, 1, 1, -8, 1, 1, 1, 1]
LPL2_3X3_ALT = np.array(LPL2_3X3_ALT).reshape((3, 3))
LPL2_3X3_ALT = (LPL2_3X3_ALT, 1)

# prebuilt gaussian kernel masks for the sake of speed
# realistically only need a few kernels anyway, so why bother calculating each time?
# each mask has already been normalized, all entries sum to 1

# kernel = 1
GAUS_3X3 = [[0.07511361, 0.1238414,  0.07511361],
            [0.1238414,  0.20417996, 0.1238414],
            [0.07511361, 0.1238414,  0.07511361]]
GAUS_3X3 = np.array(GAUS_3X3).reshape((3, 3))
# kernel = 2
GAUS_5X5 = [[0.02324684, 0.03382395, 0.03832756, 0.03382395, 0.02324684],
            [0.03382395, 0.04921356, 0.05576627, 0.04921356, 0.03382395],
            [0.03832756, 0.05576627, 0.06319146, 0.05576627, 0.03832756],
            [0.03382395, 0.04921356, 0.05576627, 0.04921356, 0.03382395],
            [0.02324684, 0.03382395, 0.03832756, 0.03382395, 0.02324684]]
GAUS_5X5 = np.array(GAUS_5X5).reshape((5, 5))
# kernel = 3
GAUS_7X7 = [[0.01129725, 0.01491455, 0.01761946, 0.01862602, 0.01761946, 0.01491455, 0.01129725],
            [0.01491455, 0.01969008, 0.02326108, 0.02458993, 0.02326108, 0.01969008, 0.01491455],
            [0.01761946, 0.02326108, 0.02747972, 0.02904957, 0.02747972, 0.02326108, 0.01761946],
            [0.01862602, 0.02458993, 0.02904957, 0.03070911, 0.02904957, 0.02458993, 0.01862602],
            [0.01761946, 0.02326108, 0.02747972, 0.02904957, 0.02747972, 0.02326108, 0.01761946],
            [0.01491455, 0.01969008, 0.02326108, 0.02458993, 0.02326108, 0.01969008, 0.01491455],
            [0.01129725, 0.01491455, 0.01761946, 0.01862602, 0.01761946, 0.01491455, 0.01129725]]
GAUS_7X7 = np.array(GAUS_7X7).reshape((7, 7))
# add additional kernel sizes up to 6(6 is max necessary) here
# for now will simply calculate them during runtime if needed
