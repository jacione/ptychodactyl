import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.misc import ascent, face
from skimage import transform
import h5py
from PIL import Image


def fft(image):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))


def ifft(image):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(image)))


def show(image):
    plt.figure(tight_layout=True)
    plt.subplot(111, xticks=[], yticks=[])
    plt.imshow(image)
    plt.show()


def random(shape, low=0.0, high=1.0):
    """
    Generates an array of random numbers between low and high,
    """
    rng = np.random.default_rng()
    return low + (high-low)*rng.random(shape)


def demo_image(size):
    """
    Generates a complex image, using one image for the amplitude and another for the phase.

    :param size: number of pixels on a side.
    :return: 2D array of shape (size, size)
    """
    x = (1024 - 768)
    f = face(gray=True)[:, x:]
    f = f / np.max(f)
    a = ascent() / np.max(ascent())
    f = transform.resize(f, (size, size), anti_aliasing=True, preserve_range=True)
    a = transform.resize(a, (size, size), anti_aliasing=True, preserve_range=True)
    img = a * np.exp(2j*np.pi*a)
    return img


def calc_error(actual, altered, border=5):
    # Calculate the difference between two complex images
    b = border
    gamma = (np.sum(actual[b:-b, b:-b] * np.conj(altered)[b:-b, b:-b]) /
             np.sum(np.abs(altered)[b:-b, b:-b] ** 2))
    error = (np.sum(np.abs(actual[b:-b, b:-b] - gamma * altered[b:-b, b:-b]) ** 2) /
             np.sum(np.abs(actual[b:-b, b:-b]) ** 2))
    return error


def convert_to_gif(pty_file):
    # arr should be a 3D numpy array, with images along the 0th axis.
    data = h5py.File(pty_file)
    images = np.array(data['data/data'])
    cx, cy = images[0].shape
    cx = cx // 2 - 30
    cy = cy // 2 + 20
    images = images[:, cx-300:cx+300, cy-300:cy+300]
    images = images - np.min(images)
    images = np.uint8(images / np.max(images) * 255)

    ims = [Image.fromarray(a) for a in images]
    ims[0].save("Data/array.gif", save_all=True, append_images=ims[1:], duration=150, loop=0)
    return


def shift(arr, shift_amt, crop=None, subpixel=False):
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
