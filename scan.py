"""
Functions to generate an array of positions for ptychographic scans
"""

import numpy as np
import utils.general as hf
from matplotlib import pyplot as plt


def xy_scan(pattern, center, width, height, step, random=True):
    """
    Generates a set of positions used for a ptychographic scan.

    :param pattern: 'rect' for rectangular, 'hex' for hexagonal, 'spiral' for spiral/orchestral
    :type pattern: str
    :param center: horizontal and vertical scan center position
    :type center: (float, float)
    :param width: scan width in millimeters
    :type width: float
    :param height: scan height in millimeters
    :type height: float
    :param step: distance between nearest-neighbor positions
    :type step: float
    :param random: if True (default), assigns a small random shift to each position
    :type random: bool
    :return: X and Y scan positions, and number of total positions
    :rtype: (np.ndarray, np.ndarray, int)
    """
    scan = STYLE_DICT[pattern]
    return scan(center, width, height, step, random)
    

def rect_scan(center, width, height, step, random):
    """
    Generates a rectangular ptychographic scan. Positions are aligned into rows and columns, such that the four nearest
    neighbors of each position are directly above, below, and to either side.

    :param width: scan width in millimeters
    :type width: float
    :param center: horizontal and vertical scan center position
    :type center: (float, float)
    :param height: scan height in millimeters
    :type height: float
    :param step: distance between nearest-neighbor positions
    :type step: float
    :param random: if True (default), assigns a small random shift to each position
    :type random: bool
    :return: X and Y scan positions, and number of total positions
    :rtype: (np.ndarray, np.ndarray, int)
    """
    x = np.arange(center[0]-width/2, center[0]+width/2, step)
    y = np.arange(center[1]-height/2, center[1]+height/2, step)
    X, Y = np.meshgrid(x, y)
    if random:
        dxy = 0.05*step
        X = X + hf.random(X.shape, -dxy, dxy)
        Y = Y + hf.random(Y.shape, -dxy, dxy)
    X = np.ravel(X)
    Y = np.ravel(Y)
    N = len(X)
    return np.round(X, 6), np.round(Y, 6), N


def hex_scan(center, width, height, step, random):
    """
    Generates a hexagonal ptychographic scan. Positions are aligned into rows and (loosely) columns, such that the six
    nearest neighbors of each position form a regular hexagon.

    :param width: scan width in millimeters
    :type width: float
    :param center: horizontal and vertical scan center position
    :type center: (float, float)
    :param height: scan height in millimeters
    :type height: float
    :param step: distance between nearest-neighbor positions
    :type step: float
    :param random: if True (default), assigns a small random shift to each position
    :type random: bool
    :return: X and Y scan positions, and number of total positions
    :rtype: (np.ndarray, np.ndarray, int)
    """
    x = np.arange(center[0]-width/2, center[0]+width/2, step)
    y = np.arange(center[1]-height/2, center[1]+height/2, step*np.sqrt(3)/2)  # The spacing between rows takes the
                                                                              # shifting into account
    X, Y = np.meshgrid(x, y)
    for i in range(X.shape[0]):
        # This shifts every other row by a half step to make the hexagonal pattern
        if i % 2:
            X[i] = X[i] + step / 2
    if random:
        dxy = 0.05*step
        X = X + hf.random(X.shape, -dxy, dxy)
        Y = Y + hf.random(Y.shape, -dxy, dxy)
    X = np.ravel(X)
    Y = np.ravel(Y)
    N = len(X)
    return np.round(X, 6), np.round(Y, 6), N


def spiral_scan(center, width, height, step, random):
    """
    Generates a spiral ptychographic scan. Positions are arranged spiraling out such that the points are spaced evenly
    along the path and the loops are spaced evenly from each other.

    :param width: scan width in millimeters
    :type width: float
    :param center: horizontal and vertical scan center position
    :type center: (float, float)
    :param height: scan height in millimeters
    :type height: float
    :param step: distance between nearest-neighbor positions
    :type step: float
    :param random: if True (default), assigns a small random shift to each position
    :type random: bool
    :return: X and Y scan positions, and number of total positions
    :rtype: (np.ndarray, np.ndarray, int)
    """
    r_max = np.sqrt(width**2 + height**2) / 2
    coords = np.array([-step/5, step/3.3])
    r = step/4
    phi = 0
    while r < r_max:
        phi = phi + (step / r)
        r = phi * step / (2*np.pi)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        coords = np.vstack((coords, [x, y]))
    X = coords[:, 0]
    Y = coords[:, 1]
    bounds = (X < width/2) * (X > -width/2) * (Y < height/2) * (Y > -height/2)
    if random:
        dxy = 0.05*step
        X = X + hf.random(X.shape, -dxy, dxy)
        Y = Y + hf.random(Y.shape, -dxy, dxy)
    X = X[bounds] + center[0]
    Y = Y[bounds] + center[1]
    N = len(X)
    return np.round(X, 6), np.round(Y, 6), N


def r_scan(num_steps=0, full_range=180):
    """
    Generates an array of rotational positions. This is basically just a fancy wrapper for np.linspace, but it includes
    a special case for non-rotational ptychographic scans.

    :param num_steps: Number of rotational positions. If zero, the return value will allow for a non-rotational
        (2D) ptychography dataset.
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
    xx, yy, _ = xy_scan('spiral', 1, 1, 0.1)
    plt.scatter(xx, yy)
    plt.gca().set_aspect('equal')
    plt.show()
