import numpy as np
from matplotlib import pyplot as plt
import h5py
import helper_funcs as hf
from abc import ABC, abstractmethod
from scipy import ndimage
from matplotlib.patches import Rectangle


class PtychoData(ABC):
    @abstractmethod
    def __init__(self, file):
        self._file = None
        self._images = None
        self._num_entries = None
        self._shape = None
        self._translation = None
        self._rotation = None
        self._pixel_size = None
        self._energy = None
        self._wavelength = None
        self._distance = None

    def __repr__(self):
        s = f'Ptycho data object:\n' \
            f'\tTotal entries: {self._num_entries}\n' \
            f'\tPixel size: {self._pixel_size} um\n' \
            f'\tEnergy: {self._energy} eV\n' \
            f'\tWavelength: {self._wavelength} um\n' \
            f'\tDistance: {self._distance} um'
        return s

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


class CXIData(PtychoData):
    def __init__(self, cxi_file):

        self._file = cxi_file
        data = h5py.File(self._file, 'r')

        self._images = np.array(data['entry_1/data_1/data'])
        self._num_entries = self._images.shape[0]
        self._shape = np.array((self._images.shape[1], self._images.shape[2]))

        self._translation = np.array(data['entry_1/data_1/translation']) * 1e6  # microns
        try:
            self._rotation = np.array(data['entry_1/data_1/rotation']) * np.pi / 180  # radians
        except KeyError:
            self._rotation = None

        self._pixel_size = np.array(data['entry_1/instrument_1/detector_1/x_pixel_size']) * 1E6  # microns
        self._energy = np.array(data['entry_1/instrument_1/source_1/energy']) / 1.602176634e-19  # electron-volts
        self._wavelength = 1.240 / self._energy  # microns
        self._distance = np.array(data['entry_1/instrument_1/detector_1/distance']) * 1e6  # microns


class Reconstruction:
    def __init__(self, data):
        self.data = data
        # print(self.data)
        self.probe = hf.init_probe(self.data.shape)
        self.crop = self.data.shape[0]
        probe_pixel_size = data.distance * data.wavelength / (data.shape[0] * data.pixel_size)
        # print(probe_pixel_size)
        # print(np.max(np.abs(self.data.translation), axis=0))
        # print(np.max(np.abs(self.data.translation), axis=0) / probe_pixel_size)
        # print(np.around(np.max(np.abs(self.data.translation), axis=0) / probe_pixel_size).astype('int'))
        max_translation = np.around(np.max(np.abs(self.data.translation), axis=0) / probe_pixel_size).astype('int')
        # print(data.shape + max_translation)
        self.object = hf.init_object(data.shape + max_translation)
        self.temp_object = np.zeros_like(self.object)
        self.rng = np.random.default_rng()
        self.probe_energy = np.max(np.sum(self.data.images, axis=(1, 2)))

        # This is the translation data IN UNITS OF PIXELS
        self.translation = self.data.translation / probe_pixel_size

        self._algs = {
            'epie': self.epie
        }

    def run(self, num_iterations, algorithm, show_progress=False):
        update_function = self._algs[algorithm]
        n = 0
        for x in range(num_iterations):
            entries = np.arange(self.data.num_entries)
            self.rng.shuffle(entries)
            for i in entries:
                update_function(i, update_probe=(x > 0))
                n += 1
                if show_progress and n == 10:
                    self.show_object_and_probe(i)
                    n = 0

    def show_progress(self, i):
        if len(plt.get_fignums()) == 0:
            kw = {'xticks': [], 'yticks': []}
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, subplot_kw=kw)
            ax1.set_title('Amplitude')
            ax2.set_title('Phase')
        else:
            [ax1, ax2, ax3, ax4] = plt.gcf().get_axes()
        ax1.imshow(np.abs(self.object), cmap='bone')
        ax1.add_artist(Rectangle(-self.translation[i].T, self.data.shape[0], self.data.shape[1], fill=False, color='r'))
        ax2.imshow(np.angle(self.object), cmap='hsv')
        ax3.imshow(np.abs(self.probe), cmap='bone')
        ax4.imshow(np.angle(self.probe), cmap='hsv')
        plt.draw()
        plt.pause(0.1)

    def show_object_and_probe(self, i=None):
        kw = {'xticks': [], 'yticks': []}
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, subplot_kw=kw)
        ax1.set_title('Amplitude')
        ax2.set_title('Phase')
        ax1.imshow(np.abs(self.object), cmap='bone')
        if i is not None:
            ax1.add_artist(
                Rectangle(-np.flip(self.translation[i]), self.data.shape[0], self.data.shape[1], fill=False, color='r'))
        ax2.imshow(np.angle(self.object), cmap='hsv')
        ax3.imshow(np.abs(self.probe), cmap='bone')
        ax4.imshow(np.angle(self.probe), cmap='hsv')
        plt.show()

    def update(self, i, probe_update, object_update):
        # Apply the probe update
        self.probe = self.probe + probe_update

        # Apply the object update
        # In order to register subpixel positions, this must be done in a few steps
        self.temp_object[:self.crop, :self.crop] = object_update
        self.temp_object = hf.shift(self.temp_object, -self.translation[i])
        self.object = self.object + self.temp_object

        self.temp_object = np.zeros_like(self.object)

    def epie(self, i, alpha=0.8, beta=0.2, update_probe=True):
        # Take a slice of the object that's the same size as the probe
        region = hf.shift(self.object, self.translation[i], crop=self.data.shape[0])
        psi1 = self.probe * region
        PSI1 = np.fft.fftn(psi1)
        PSI2 = self.data.images[i] * PSI1 / np.abs(PSI1)
        psi2 = np.fft.ifftn(PSI2)
        d_psi = psi2 - psi1

        self.probe = self.probe * np.sqrt(self.probe_energy/np.sum(np.abs(self.probe)**2)) / self.data.shape[0]

        object_update = alpha * d_psi * np.conj(self.probe) / np.max(np.abs(self.probe)) ** 2
        if update_probe:
            probe_update = beta * d_psi * np.conj(region) / np.max(np.abs(region)) ** 2
        else:
            probe_update = 0

        self.update(i, probe_update, object_update)
