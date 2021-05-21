"""
Class for ptycho data sets
"""

import numpy as np
from matplotlib import pyplot as plt
import os
import h5py
from datetime import date


class DataSet:
    def __init__(self, num_takes, title, directory, im_shape, pixel_size, distance, energy, is2d, verbose=False):
        # General parameters
        self.verbose = verbose
        self.N = num_takes  # Total number of takes (all translations and rotations)
        self.__n = 0  # Current take
        self.im_shape = im_shape  # image resolution
        self.position = np.empty((self.N, 4))  # (x, y, z, th) for each get_position
        self.is2d = is2d

        self.data = np.empty((self.N, self.im_shape[1], self.im_shape[0]), dtype='i2')

        # These parameters probably won't change
        self.energy = energy * 1.602176634e-19  # photon energy in JOULES
        self.pixel_size = pixel_size * 1e-6  # pixel side length in meters
        self.distance = distance  # sample-to-detector distance in meters

        # Parameters for saving the data afterward
        self.dir = directory
        self.title = title + '-'
        return

    def print(self, text):
        if self.verbose:
            print(text)

    def record_data(self, position, im_data):
        self.position[self.__n] = position * 1e-3  # Record position in meters
        self.data[self.__n] = im_data
        self.print(f'Take {self.__n} recorded!')
        self.__n += 1
        return

    def show(self, n=None):
        if n is None:
            n = self.__n
        plt.subplot(111, xticks=[], yticks=[])
        plt.tight_layout()
        plt.imshow(self.data[n], cmap='plasma')

    def save_to_cxi(self):
        if self.__n != self.N:
            if input('Scan not complete! Are you sure you want to save? (Y/n)').lower() == 'n':
                return
            else:
                self.title = self.title + 'INCOMPLETE'
        else:
            self.title = self.title + date.today().isoformat()
        print(f'Saving data as {self.dir}/{self.title}.cxi')

        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

        # Create file
        f = h5py.File(f'{self.dir}/{self.title}.cxi', 'w')

        # Create entry
        entry = f.create_group('entry_1')

        # Save collected data
        data = entry.create_group('data_1')
        data.create_dataset('data', data=self.data, dtype='i2')
        if self.is2d:
            data.create_dataset('translation', data=self.position[:, :2])
        else:
            data.create_dataset('translation', data=self.position)

        # Save experimental parameters
        inst = entry.create_group('instrument_1')
        source = inst.create_group('source_1')
        source.create_dataset('energy', data=self.energy)
        detector = inst.create_group('detector_1')
        detector.create_dataset('distance', data=self.distance)
        detector.create_dataset('x_pixel_size', data=self.pixel_size)
        detector.create_dataset('y_pixel_size', data=self.pixel_size)

        f.close()
        return
