import numpy as np
from matplotlib import pyplot as plt
import h5py


class CXIData:
    def __init__(self, cxi_file):
        """

        :param cxi_file:
        """
        self.__file = cxi_file
        data = h5py.File(self.__file, 'r')

        self.__data = np.array(data['entry_1/data_1/data'])
        self.__num_entries = self.__data.shape[0]

        self.__translation = np.array(data['entry_1/data_1/translation'])

        self.__pixel_size = np.around(np.array(data['entry_1/instrument_1/detector_1/x_pixel_size']) * 1E6)
        self.__energy = np.array(data['entry_1/instrument_1/source_1/energy']) / 1.602176634e-19  # electron-volts
        self.__wavelength = 1240 / self.__energy  # nanometers
        self.__distance = np.array(data['entry_1/instrument_1/detector_1/distance'])  # meters

    def data(self, n):
        return self.__data[n]

    def trans(self, n):
        return self.__translation[n]

    def pixel_size(self):
        return self.__pixel_size

    def energy(self):
        return self.__energy

    def wavelength(self):
        return self.__wavelength

    def distance(self):
        return self.__distance
