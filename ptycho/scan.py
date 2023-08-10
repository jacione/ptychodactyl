"""
Functions to generate an array of positions for ptychographic scans
"""

import numpy as np
import ptycho.utils as hf
from matplotlib import pyplot as plt


def xy_scan(config):
    """
    Generates a set of positions used for a ptychographic scan.

    :param config: configuration dictionary
    :type config: dict
    :return: X and Y scan positions, and number of total positions
    :rtype: (np.ndarray, np.ndarray, int)
    """
    scan = STYLE_DICT[config["scan_pattern"]]
    return scan(**config)
    

def rect_scan(scan_center, scan_width, scan_height, scan_step, random_shift, **_):
    """
    Generates a rectangular ptychographic scan. Positions are aligned into rows and columns, such that the four nearest
    neighbors of each position are directly above, below, and to either side.

    :param scan_width: scan width in millimeters
    :type scan_width: float
    :param scan_center: horizontal and vertical scan center position
    :type scan_center: (float, float)
    :param scan_height: scan height in millimeters
    :type scan_height: float
    :param scan_step: distance between nearest-neighbor positions
    :type scan_step: float
    :param random_shift: if True (default), assigns a small random shift to each position
    :type random_shift: bool
    :return: X and Y scan positions, and number of total positions
    :rtype: (np.ndarray, np.ndarray, int)
    """
    x = np.arange(scan_center[0] - scan_width / 2, scan_center[0] + scan_width / 2, scan_step)
    y = np.arange(scan_center[1] - scan_height / 2, scan_center[1] + scan_height / 2, scan_step)
    X, Y = np.meshgrid(x, y)
    for i in range(x.shape[0]):
        # This flips every other row so that the final scan is a snake-like pattern rather than always starting from
        # one side.
        if i % 2:
            X[i] = np.flip(X[i])
    if random_shift:
        dxy = 0.05 * scan_step
        X = X + hf.random(X.shape, -dxy, dxy)
        Y = Y + hf.random(Y.shape, -dxy, dxy)
    X = np.ravel(X)
    Y = np.ravel(Y)
    N = len(X)
    return np.round(X, 6), np.round(Y, 6), N


def hex_scan(scan_center, scan_width, scan_height, scan_step, random_shift, **_):
    """
    Generates a hexagonal ptychographic scan. Positions are aligned into rows and (loosely) columns, such that the six
    nearest neighbors of each position form a regular hexagon.

    :param scan_width: scan width in millimeters
    :type scan_width: float
    :param scan_center: horizontal and vertical scan center position
    :type scan_center: (float, float)
    :param scan_height: scan height in millimeters
    :type scan_height: float
    :param scan_step: distance between nearest-neighbor positions
    :type scan_step: float
    :param random_shift: if True (default), assigns a small random shift to each position
    :type random_shift: bool
    :return: X and Y scan positions, and number of total positions
    :rtype: (np.ndarray, np.ndarray, int)
    """
    x = np.arange(scan_center[0] - scan_width / 2, scan_center[0] + scan_width / 2, scan_step)
    y = np.arange(scan_center[1] - scan_height / 2, scan_center[1] + scan_height / 2, scan_step * np.sqrt(3) / 2)
    # The spacing between rows takes the shifting into account
    X, Y = np.meshgrid(x, y)
    for i in range(X.shape[0]):
        # This shifts every other row by a half step to make the hexagonal pattern and flips every other row to make a
        # snake like pattern
        if i % 2:
            X[i] = np.flip(X[i] + scan_step / 2)

    if random_shift:
        dxy = 0.05 * scan_step
        X = X + hf.random(X.shape, -dxy, dxy)
        Y = Y + hf.random(Y.shape, -dxy, dxy)
    X = np.ravel(X)
    Y = np.ravel(Y)
    N = len(X)
    return np.round(X, 6), np.round(Y, 6), N


def spiral_scan(scan_center, scan_width, scan_height, scan_step, random_shift, **_):
    """
    Generates a spiral ptychographic scan. Positions are arranged spiraling out such that the points are spaced evenly
    along the path and the loops are spaced evenly from each other.

    :param scan_width: scan width in millimeters
    :type scan_width: float
    :param scan_center: horizontal and vertical scan center position
    :type scan_center: (float, float)
    :param scan_height: scan height in millimeters
    :type scan_height: float
    :param scan_step: distance between nearest-neighbor positions
    :type scan_step: float
    :param random_shift: if True (default), assigns a small random shift to each position
    :type random_shift: bool
    :return: X and Y scan positions, and number of total positions
    :rtype: (np.ndarray, np.ndarray, int)
    """
    r_max = np.sqrt(scan_width ** 2 + scan_height ** 2) / 2
    coords = np.array([-scan_step / 5, scan_step / 3.3])
    r = scan_step / 4
    phi = 0
    while r < r_max:
        phi = phi + (scan_step / r)
        r = phi * scan_step / (2 * np.pi)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        coords = np.vstack((coords, [x, y]))
    X = coords[:, 0]
    Y = coords[:, 1]
    bounds = (X < scan_width / 2) * (X > -scan_width / 2) * (Y < scan_height / 2) * (Y > -scan_height / 2)
    if random_shift:
        dxy = 0.05 * scan_step
        X = X + hf.random(X.shape, -dxy, dxy)
        Y = Y + hf.random(Y.shape, -dxy, dxy)
    X = X[bounds] + scan_center[0]
    Y = Y[bounds] + scan_center[1]
    N = len(X)
    return np.round(X, 6), np.round(Y, 6), N


def r_scan(num_steps=0, full_range=180, **_):
    """
    Generates an array of rotational positions. This is basically just a fancy wrapper for np.linspace, but it includes
    a special case for non-rotational ptychographic scans.

    :param num_steps: Number of rotational positions. If zero, the return value will allow for a non-rotational
        (2D) ptycho dataset.
    :type num_steps: int
    :param full_range: Full range of angles to sweep through.
    :type full_range: float
    :return: 1D array of angles
    :rtype: np.ndarray
    """
    if num_steps == 0:
        return np.array([0]), 1
    else:
        return np.linspace(-full_range/2, full_range/2, num_steps), num_steps


STYLE_DICT = {'rect': rect_scan, 'hex': hex_scan, 'spiral': spiral_scan}


if __name__ == '__main__':
    a = 0
    xx, yy, _ = spiral_scan((0, 0), 3, 3, 0.1, True)
    plt.scatter(xx, yy)
    plt.gca().set_aspect('equal')
    plt.show()
