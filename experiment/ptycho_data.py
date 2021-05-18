"""
Class for ptycho data sets
"""

import numpy as np
from matplotlib import pyplot as plt
import os
import h5py
from datetime import datetime


class DataSet:
    def __init__(self, num_takes, title, directory, im_shape, distance, energy):
        # General parameters
        self.N = num_takes  # Total number of takes (all translations and rotations)
        self.__n = 0  # Current take
        self.im_shape = im_shape  # image resolution
        self.position = np.empty((self.N, 3))  # (x, y, th) for each position

        self.data = np.empty((self.N, self.im_shape[0], self.im_shape[1]), dtype='i2')

        # These parameters probably won't change
        self.energy = energy  # photon energy in electron-volts
        self.pixel_size = 2.2  # pixel side length in micrometers
        self.distance = distance  # sample-to-detector distance in meters

        # Parameters for saving the data afterward
        self.dir = directory
        self.title = title
        return

    def record_data(self, position, im_data):
        self.position[self.__n] = position
        self.data[self.__n] = im_data
        self.__n += 1
        if self.__n == self.N:
            self.title = self.title + datetime.now().isoformat(sep=' ', timespec='seconds')
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
                self.title = self.title + ' (INCOMPLETE)'
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        f = h5py.File(f'{self.dir}/{self.title}.cxi', 'w')

        entry = f.create_group('entry_1')

        data = entry.create_group('data_1')
        data.create_dataset('data', data=self.data, dtype='int')
        data.create_dataset('translation', data=self.position)

        inst = entry.create_group('instrument_1')
        source = inst.create_group('source_1')
        source.create_dataset('energy', data=self.energy)
        detector = inst.create_group('detector_1')
        detector.create_dataset('distance', data=self.distance)
        detector.create_dataset('x_pixel_size', data=self.pixel_size)
        detector.create_dataset('y_pixel_size', data=self.pixel_size)

        f.close()
        return
