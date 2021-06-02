"""
Class for ptycho data sets
"""

import numpy as np
from matplotlib import pyplot as plt
import os
import h5py
from datetime import date
from skimage import draw, transform
from scipy import ndimage
from scipy.misc import ascent, face
import helper_funcs as hf
from matplotlib.patches import Circle


class DataSet:
    def __init__(self, num_takes, title, directory, im_shape, pixel_size, distance, energy, is3d, verbose=False):
        # General parameters
        self.verbose = verbose
        self.num_entries = num_takes  # Total number of takes (all translations and rotations)
        self._n = 0  # Current take
        self.im_shape = im_shape  # image resolution
        self.translation = np.empty((self.num_entries, 2))
        self.rotation = np.empty(self.num_entries)
        self.is3d = is3d

        self.data = np.empty((self.num_entries, self.im_shape[1], self.im_shape[0]), dtype='u2')

        # These parameters probably won't change
        self.energy = energy * 1.602176634e-19  # photon energy in JOULES
        self.wavelength = 1.240 * 1e-6 / energy  # wavelength in meters
        self.pixel_size = pixel_size * 1e-6  # pixel side length in meters
        self.distance = distance  # sample-to-detector distance in meters

        # Parameters for saving the data afterward
        self.dir = directory
        self.title = title
        self.is_finalized = False
        return

    def __repr__(self):
        s = f'Ptycho data object:\n' \
            f'\tTotal entries: {self.num_entries}\n' \
            f'\tPixel size: {self.pixel_size} m\n' \
            f'\tEnergy: {self.energy} J\n' \
            f'\tWavelength: {self.wavelength} m\n' \
            f'\tDistance: {self.distance} m'
        return s

    def print(self, text):
        if self.verbose:
            print(text)

    def record_data(self, position, im_data):
        self.translation[self._n] = position[:2] * 1e-3  # Receive in millimeters, record in meters
        if self.is3d:
            self.rotation[self._n] = position[3]
        self.data[self._n] = im_data
        self.print(f'Take {self._n} of {self.num_entries} recorded!')
        self._n += 1
        return

    def finalize(self, timestamp=True):
        if not self.is_finalized:
            self.translation = self.translation - np.max(self.translation, axis=0)
            if timestamp:
                self.title = self.title + '-' + date.today().isoformat()
            self.is_finalized = True
        return

    def show(self, n=None):
        if n is None:
            n = self._n
        plt.subplot(121, xticks=[], yticks=[], title='Linear')
        plt.imshow(self.data[n], cmap='plasma')
        plt.subplot(122, xticks=[], yticks=[], title='Logarithmic')
        plt.imshow(np.log(self.data[n]+1), cmap='plasma')
        plt.tight_layout()
        plt.show()

    def save_to_cxi(self, timestamp=True):
        if self._n == self.num_entries:
            self.finalize(timestamp)
        else:
            if input('Scan not complete! Are you sure you want to save? (Y/n)').lower() == 'n':
                return
            else:
                self.title = self.title + '-INCOMPLETE'

        print(f'Saving data as {self.dir}/{self.title}.cxi')

        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

        # Create file
        f = h5py.File(f'{self.dir}/{self.title}.cxi', 'w')

        # Create entry
        entry = f.create_group('entry_1')

        # Save collected data
        data = entry.create_group('data_1')
        data.create_dataset('data', data=self.data, dtype='u2')
        data.create_dataset('translation', data=self.translation)
        if self.is3d:
            data.create_dataset('rotation', data=self.rotation)

        # Save experimental parameters
        inst = entry.create_group('instrument_1')
        source = inst.create_group('source_1')
        source.create_dataset('energy', data=self.energy)
        detector = inst.create_group('detector_1')
        detector.create_dataset('distance', data=self.distance)
        detector.create_dataset('x_pixel_size', data=self.pixel_size, dtype='float')
        detector.create_dataset('y_pixel_size', data=self.pixel_size, dtype='float')

        f.close()
        return


class GeneratedData(DataSet):
    def __init__(self, ims_per_line, max_shift, title, directory, im_size=2 ** 7):
        super().__init__(ims_per_line**2, title, directory, (im_size, im_size), 5.5, 0.1, 2.04, True, True)
        self.data = None
        self.data = np.empty((self.num_entries, self.im_shape[1], self.im_shape[0]))

        max_shift = max_shift

        # Generate a basic probe
        probe_pixel_size = self.distance * self.wavelength / (im_size * self.pixel_size)
        print(probe_pixel_size)
        probe_array = np.zeros((im_size, im_size))
        probe_center = im_size / 2
        probe_radius = .0025 / probe_pixel_size
        rr, cc = draw.disk((probe_center, probe_center), probe_radius)
        probe_array[rr, cc] = 1
        probe_array = ndimage.gaussian_filter(probe_array, 3)

        x = np.linspace(0, max_shift/probe_pixel_size, ims_per_line)
        X, Y = np.meshgrid(x, x)
        X = np.ravel(X) + hf.random(self.num_entries, -2, 2)
        Y = np.ravel(Y) + hf.random(self.num_entries, -2, 2)

        # Generate an object
        obj_size = im_size + int(max_shift/probe_pixel_size) + 1
        print(obj_size)
        img = ascent()
        object_array = transform.resize(img, (obj_size, obj_size), anti_aliasing=True, preserve_range=True)

        ax = plt.subplot(111, xticks=[], yticks=[])
        ax.imshow(np.abs(object_array))
        for i in range(self.num_entries):
            ax.add_artist(Circle((probe_center+X[i], probe_center+Y[i]), probe_radius, color='r', fill=False))
        plt.show()
        quit()

        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        for i in range(self.num_entries):
            position = np.array([-X[i], -Y[i], 0, 0])
            shifted_object = hf.shift(np.copy(object_array), position[:2], crop=im_size)
            exit_wave = probe_array * shifted_object
            diffraction = np.abs(hf.fft(exit_wave))**2
            # ax1.clear()
            # ax1.imshow(np.abs(object_array))
            # ax1.add_artist(Circle((probe_center + Y[i], probe_center + X[i]), probe_radius, color='r', fill=False))
            # ax2.imshow(np.abs(exit_wave))
            # ax3.imshow(np.log(diffraction))
            # plt.draw()
            # plt.pause(0.1)
            self.record_data(position*1e3*probe_pixel_size, diffraction)

        # self.data = hf.detect(self.data, 0.9, 14)
        # self.show()
        print(self)

        self.save_to_cxi(timestamp=False)


if __name__ == '__main__':
    data = GeneratedData(12, .01, 'fake', 'libs/dummy_data')
