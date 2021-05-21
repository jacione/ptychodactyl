"""
Measure the beamwidth on the camera
"""

import numpy as np
import mightex
from scipy.ndimage import center_of_mass
import click


@click.command()
@click.option('-p', '--print', 'disp', is_flag=True, default=False)
def measure(disp):
    pixel_size = 2.2e-3
    with mightex.Camera() as cam:
        img = cam.get_frame()
    # img = np.ones((10, 10))
    ctr = center_of_mass(img)
    X = img[int(ctr[0])]
    Y = img[int(ctr[1])]
    half_max = np.max(img) / 2
    fw_x = len(X[X > half_max])
    fw_y = len(Y[Y > half_max])
    fwhm = (fw_x + fw_y) * pixel_size / 2
    if disp:
        print(fwhm)
    return fwhm


if __name__ == '__main__':
    measure()