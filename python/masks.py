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