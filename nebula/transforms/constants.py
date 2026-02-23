import numpy as np

########
# WGS-84 ellipsoid constants
#
WGS84_A = 6378137.0
WGS84_B = 6356752.314245179
WGS84_A2 = WGS84_A * WGS84_A
WGS84_B2 = WGS84_B * WGS84_B
WGS84_B2_OVER_A2 = WGS84_B2 / WGS84_A2
# e^2 = 1 - (b^2/a^2)
WGS84_E2 = 1.0 - WGS84_B2_OVER_A2
# e'^2 = (a^2 - b^2) / b^2
WGS84_EP2 = (WGS84_A2 - WGS84_B2) / WGS84_B2


#######
# Conversion factors
#
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


#######
# Pi constants
#
PI = np.pi
TWO_PI = 2.0 * np.pi
HALF_PI = 0.5 * np.pi
