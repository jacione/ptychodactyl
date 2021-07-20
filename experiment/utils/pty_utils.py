import tkinter as tk
from tkinter import filedialog
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


IMG_KWARGS = {'xticks': [], 'yticks': []}


def view_pty():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    f = h5py.File(file_path)
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


if __name__ == '__main__':
    view_pty()
