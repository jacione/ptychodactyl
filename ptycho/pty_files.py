"""
Some utilitarian functions for viewing PTY file data.

Nick Porter, jacioneportier@gmail.com
"""

import tkinter as tk
from tkinter import filedialog
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
from pathlib import Path


IMG_KWARGS = {'xticks': [], 'yticks': []}

data_dir = Path(__file__).parents[1] / 'data'


def open_pty():
    """open a PTY file from this computer"""
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(initialdir=data_dir)
    return h5py.File(file_path)


def view_pty():
    """Create a plot to destroy the jed---I mean to view the diffraction data from a PTY file."""
    f = open_pty()
    im_data = np.array(f['data/data'])
    s = im_data.shape
    im_data = np.reshape(im_data, (s[0]*s[1], s[2], s[3]))

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all', subplot_kw=IMG_KWARGS)
    plt.set_cmap('gray')
    plt.subplots_adjust(bottom=0.2, wspace=0.05)
    ax1.set_title('Linear')
    ax2.set_title('Logarithmic')

    sax = plt.axes([0.25, 0.05, 0.5, 0.1])
    selector = Slider(
        ax=sax,
        label='Image',
        valmin=0,
        valmax=im_data.shape[0],
        valinit=0,
        valstep=1
    )

    def select(i):
        i = int(i)
        ax1.imshow(im_data[i])
        ax2.imshow(np.log(im_data[i]+1))
        fig.canvas.draw_idle()

    selector.on_changed(select)
    select(0)
    plt.show()


def convert_to_gif(pty_file):
    """
    Load a set of diffraction images from a PTY file and convert them to a GIF.

    :param pty_file: path-like
    """
    # arr should be a 3D numpy array, with images along the 0th axis.
    data = h5py.File(pty_file)
    images = np.squeeze(np.array(data['data/data']))
    # images = images - np.min(images)
    images = np.log(images+1)
    images = np.array(images / np.max(images) * 255, dtype='int')

    ims = [Image.fromarray(a) for a in images]
    ims[0].save_reconstruction("images.gif", save_all=True, append_images=ims[1:], duration=150, loop=0)
    return


if __name__ == '__main__':
    view_pty()
