import sys
import math as m
import numpy as np
np.random.seed(123)


def cart2sph(x, y, z):

    x2_y2 = x**2 + y**2
    r = m.sqrt(x2_y2 + z**2)                    # r
    elev = m.atan2(z, m.sqrt(x2_y2))            # Elevation
    az = m.atan2(y, x)                          # Azimuthal
    
    return r, elev, az

def pol2cart(theta, rho):
    
    return rho * m.cos(theta), rho * m.sin(theta)


def azim_proj(pos):

    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)
