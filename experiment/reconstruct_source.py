import numpy as np
from matplotlib import pyplot as plt
import h5py
import helper_funcs as hf
from abc import ABC, abstractmethod
from scipy import ndimage


class Data(ABC):
    @abstractmethod
    def __init__(self, file):
        self._file = None
        self._images = []
        self._num_entries = 0
        self._shape = [0, 0]
        self._translation = []
        self._pixel_size = None
        self._energy = None
        self._wavelength = None
        self._distance = None

    @property
    def images(self):
        return self._images

    @property
    def num_entries(self):
        return self._num_entries

    @property
    def shape(self):
        return self._shape

    @property
    def translation(self):
        return self._translation

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def energy(self):
        return self._energy

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def distance(self):
        return self._distance

    def show_pattern(self, n):
        hf.show(self._images[n])


class CXIData(Data):
    def __init__(self, cxi_file):
        """

        :param cxi_file:
        """
        self._file = cxi_file
        data = h5py.File(self._file, 'r')

        self._images = np.array(data['entry_1/data_1/data'])
        self._num_entries = self._images.shape[0]
        self._shape = np.array((self._images.shape[1], self._images.shape[2]))

        self._translation = np.array(data['entry_1/data_1/translation']) * 1e6  # microns

        self._pixel_size = np.around(np.array(data['entry_1/instrument_1/detector_1/x_pixel_size']) * 1E6)  # microns
        self._energy = np.array(data['entry_1/instrument_1/source_1/energy']) / 1.602176634e-19  # electron-volts
        self._wavelength = 1.240 / self._energy  # microns
        self._distance = np.array(data['entry_1/instrument_1/detector_1/distance'] * 1e6)  # microns


class Reconstruction:
    def __init__(self, data):
        self.data = data
        self.probe = np.zeros(data.shape) + 0j
        self.crop = self.data.shape[0]
        probe_pixel_size = data.distance * data.wavelength / (data.shape[0] * data.pixel_size)
        max_translation = int(np.max(np.abs(self.data.translation), axis=0) * probe_pixel_size + 1)
        self.object = np.zeros(data.shape + max_translation) + 0.5 + 0j
        self.temp_object = np.zeros_like(self.object)
        self.rng = np.random.default_rng()
        self.probe_energy = np.max(np.sum(self.data.images, axis=(1, 2)))

        # This is the translation data IN UNITS OF PIXELS
        self.translation = self.data.translation / probe_pixel_size

    def run(self, num_iterations, algorithm):
        ALGORITHMS = {
            'epie': self.epie
        }
        update_function = ALGORITHMS[algorithm]
        for x in range(num_iterations):
            update_function()

    def update(self, probe_update, object_update, translation):
        # Apply the probe update
        self.probe = self.probe + probe_update

        # Apply the object update
        # In order to register subpixel positions, this must be done in a few steps
        self.temp_object[:self.crop, :self.crop] = object_update
        self.temp_object = hf.shift(self.temp_object, -translation)
        self.object = self.object + self.temp_object

        self.temp_object = np.zeros_like(self.object)

    def epie(self, alpha=1.0, beta=1.0):
        entries = np.arange(self.data.num_entries)
        self.rng.shuffle(entries)
        for i in entries:
            # Take a slice of the object that's the same size as the probe
            region = ndimage.shift(self.object, self.translation[i], crop=self.data.shape[0])
            psi1 = self.probe.profile * region
            PSI1 = np.fft.fftn(psi1)
            PSI2 = self.data.images[i] * PSI1 / np.abs(PSI1)
            psi2 = np.fft.ifftn(PSI2)
            d_psi = psi2 - psi1

            self.probe = self.probe * np.sqrt(self.probe_energy/np.sum(np.abs(self.probe)**2)) / self.data.shape[0]

            object_update = alpha * d_psi * np.conj(self.probe) / np.max(np.abs(self.probe)) ** 2
            probe_update = beta * d_psi * np.conj(region) / np.max(np.abs(region)) ** 2

            self.update(probe_update, object_update, self.translation[i])
