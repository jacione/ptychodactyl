"""
Some utilitarian functions to help simplify the rest of the modules.
"""
import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.misc import ascent, face
from skimage import transform, draw
from skimage import data as skidata


def fft(image):
    """Performs a fast-fourier transform with correct shifting"""
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))


def ifft(image):
    """Performs an inverse fast-fourier transform with correct shifting"""
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(image)))


def random(shape, low=0.0, high=1.0):
    """
    Generate an array of random values within a certain range.

    :param shape: tuple of ints, dimensions of the output array
    :type shape: tuple
    :param low: lower bound
    :type low: float
    :param high: upper bound
    :type high: float
    :return: array of random values
    :rtype: np.ndarray
    """
    rng = np.random.default_rng()
    return low + (high-low)*rng.random(shape)


def shift(arr, shift_amt, crop=None, subpixel=False):
    """
    Shift an image array by a given amount.

    :param arr: 2D image array that will be shifted
    :type arr: np.ndarray
    :param shift_amt: horizontal and vertical shift amount in pixels
    :type shift_amt: (float, float)
    :param crop: optional size to crop the shifted image
    :type crop: int
    :param subpixel: if True, image will be shifted by the exact shift amount using subpixel registration (higher
        precision). If False, shift amount will be rounded to an integer value (higher speed). Default is False.
    :type subpixel: bool
    :return: shifted copy of the input array.
    :rtype: np.ndarray
    """
    if subpixel:
        amp = ndimage.shift(np.abs(arr), shift_amt)
        phi = np.angle(ndimage.shift(arr, shift_amt))
        new_arr = amp * np.exp(1j * phi)
    else:
        x = int(shift_amt[0] + 0.5)
        y = int(shift_amt[1] + 0.5)
        new_arr = np.roll(arr, (x, y), axis=(0, 1))
    if crop is not None:
        new_arr = new_arr[:crop, :crop]
    return new_arr


def normalize(arr):
    """
    Project an array onto the range 0 to 1.

    :param arr: any size array
    :type arr: np.ndarray
    :return: normalized array
    :rtype: np.ndarray
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def calc_error(actual, altered, border=5):
    """
    Calculate the difference between two complex images. Taken from the ePIE paper (Maiden 2009)

    :param actual: 2D array, the known correct image.
    :type actual: np.ndarray
    :param altered: 2D array, the image for which error is being calculated.
    :type altered: np.ndarray
    :param border: number of pixels on each edge to ignore when calculating the error.
    :type border: int
    :return: an error value
    :rtype: float
    """
    b = border
    gamma = (np.sum(actual[b:-b, b:-b] * np.conj(altered)[b:-b, b:-b]) /
             np.sum(np.abs(altered)[b:-b, b:-b] ** 2))
    error = (np.sum(np.abs(actual[b:-b, b:-b] - gamma * altered[b:-b, b:-b]) ** 2) /
             np.sum(np.abs(actual[b:-b, b:-b]) ** 2))
    return error


def demo_binary(size):
    """
    Generate a complex square image with binary amplitude.

    :param size: number of pixels on a side
    :type size: int
    :return: 2D complex array
    :rtype: np.ndarray
    """
    arr = np.zeros((size, size))+0j

    # Draw horizontal and vertical bars
    num_bars = 7
    b_start = 0.1*size
    b_end = 0.45*size
    bar, width = np.linspace(b_start, b_end, num_bars+1, retstep=True)
    for i in range(num_bars):
        rr, cc = np.array(draw.rectangle((bar[i], b_start), (bar[i]+0.6*width, b_end)), dtype='int')
        arr[rr, cc] = 1 * np.exp(1j*np.pi*(i+1)/4)
        arr = np.rot90(arr, k=2)
        rr, cc = np.array(draw.rectangle((b_start, bar[i]), (b_end, bar[i]+0.6*width)), dtype='int')
        arr[rr, cc] = 1 * np.exp(1j*np.pi*(i+1)/4)
        arr = np.rot90(arr, k=2)

    # Draw some concentric circles
    c_ctr = (0.3*size, 0.7*size)
    c_radii, dr = np.linspace(0.2*size, 0, 9, retstep=True)
    val = 1
    for c_radius in c_radii:
        arr[draw.disk(c_ctr, c_radius)] = val * np.exp(-4j*np.pi*c_radius/size)
        val = int(not val)

    # Draw a silhouette of a horse
    h_size = int(0.4*size)
    horse = transform.resize(np.logical_not(skidata.horse()), (h_size, h_size))
    h_col = int(0.075*size)
    h_row = int(0.525*size)
    arr[h_row:h_row+h_size, h_col:h_col+h_size] = horse * np.exp(-1j*horse)

    return arr


def demo_image(size):
    """
    Generate a complex square image, using one image for the amplitude and another for the phase.

    :param size: number of pixels on a side
    :type size: int
    :return: 2D complex array
    :rtype: np.ndarray
    """
    x = (1024 - 768)
    f = face(gray=True)[:, x:]
    f = f / np.max(f)
    a = ascent() / np.max(ascent())
    f = transform.resize(f, (size, size), anti_aliasing=True, preserve_range=True)
    a = transform.resize(a, (size, size), anti_aliasing=True, preserve_range=True)
    img = a * np.exp(2j*np.pi*a)
    return img


def parse_specs(filename):
    """
    Parses a spec file into a dictionary. Only tested with TXT files.

    :param filename: name of the file to parse. Must be in the main directory (3d-ptycho).
    :type filename: str
    :return: A dictionary containing all the desired parameters as key-value pairs.
    :rtype: dict
    """
    parents = Path(__file__).parents
    for p in parents:
        try:
            os.chdir(p)
            file = open(filename, 'r')
            break
        except FileNotFoundError:
            continue
    else:
        raise FileNotFoundError(f'Could not find spec_file called "{filename}"')

    specs = {}  # create an empty dictionary

    def sub_parse(s):
        # Assigns an appropriate type to the value strings taken from a spec file.
        s = s.strip()
        if 'true' in s.lower():
            s = True
        elif 'false' in s.lower():
            s = False
        else:
            try:
                if '.' in s:
                    s = float(s)
                else:
                    s = int(s)
            except ValueError:
                pass
        return s

    for line in file.readlines():
        line, _, _ = line.partition('#')  # Separate the comments from the actual values
        if "=" in line:
            key, val = map(str.strip, line.split("="))
            if ',' in val:
                val = tuple(map(sub_parse, val.split(',')))
            else:
                val = sub_parse(val)
            specs[key] = val
    try:
        specs['num_stages'] = len(specs['algorithm'])
    except KeyError:
        pass
    return specs


if __name__ == '__main__':
    a = demo_binary(512)
    plt.subplot(121)
    plt.imshow(np.abs(a), cmap='gray')
    plt.subplot(122)
    plt.imshow(np.angle(a), cmap='hsv', clim=[-np.pi, np.pi])
    plt.show()
