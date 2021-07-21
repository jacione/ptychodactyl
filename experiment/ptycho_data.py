"""
Class definitions for ptychographic datasets.

Nick Porter, jacioneportier@gmail.com
"""

import numpy as np
from matplotlib import pyplot as plt
import h5py
from datetime import date
from skimage import draw
from scipy import ndimage
from abc import ABC, abstractmethod
from matplotlib.patches import Circle
from matplotlib.animation import ArtistAnimation
import progressbar
import os

import experiment.utils.helper_funcs as hf
from experiment.scan import xy_scan


class PtychoData(ABC):
    """
    Abstract parent class for all ptychographic datasets.
    """
    @abstractmethod
    def __init__(self):
        self._file = None
        self._im_data = None
        self._bkgd = None
        self._num_entries = None
        self._num_translations = None
        self._num_rotations = None
        self._shape = None
        self._position = None
        self._pixel_size = None
        self._obj_pixel_size = None
        self._energy = None
        self._wavelength = None
        self._distance = None

    def __repr__(self):
        s = f'Ptycho data object:\n' \
            f'\tTotal entries: {self.num_entries}\n' \
            f'\tImg pixel size: {np.round(self.pixel_size, 3)} um\n' \
            f'\tObj pixel size: {np.round(self.obj_pixel_size, 3)} um\n' \
            f'\tEnergy: {np.round(self.energy, 3)} eV\n' \
            f'\tWavelength: {np.round(self.wavelength, 4)} um\n' \
            f'\tDistance: {self.distance} um\n' \
            f'\tMax probe size: {np.round(self.distance*self.wavelength/(2*self.pixel_size),0)} um'
        return s

    @property
    def file(self):
        return self._file

    @property
    def im_data(self):
        return self._im_data

    @property
    def bkgd(self):
        return self._bkgd

    @property
    def num_entries(self):
        return self._num_entries

    @property
    def num_translations(self):
        return self._num_translations

    @property
    def num_rotations(self):
        return self._num_rotations

    @property
    def shape(self):
        return self._shape

    @property
    def position(self):
        return self._position

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def obj_pixel_size(self):
        return self._obj_pixel_size

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
        hf.show(self._im_data[n])


class LoadData(PtychoData):
    """
    PtychoData subclass for data loaded from a *.pty file.
    """
    def __init__(self, pty_file: str, flip_images='', flip_positions='', background_subtract=True, vbleed_correct=0.0,
                 threshold=0.0):
        super().__init__()

        # LOAD DATA FROM FILE #########################################################################################
        # Open the .pty file
        self._file = pty_file
        os.chdir(os.path.dirname(__file__))
        os.chdir('../data')
        f = h5py.File(self._file, 'r')

        # Load diffraction image data
        self._im_data = np.array(f['data/data'])
        shape = self.im_data.shape
        self._num_rotations = shape[0]
        self._num_translations = shape[1]
        self._num_entries = shape[0] * shape[1]
        self._shape = np.array((shape[2], shape[3]))

        # Load background image data
        self._bkgd = np.array(f['data/background'])

        # Load position data
        self._position = np.array(f['data/position'])  # millimeters / degrees
        self._position[:, :, :2] = self._position[:, :, :2] * 1e3  # Convert to microns / degrees

        # Load other experiment parameters
        self._energy = np.array(f['equipment/source/energy'])  # electron-volts
        self._wavelength = np.array(f['equipment/source/wavelength'])  # microns
        self._distance = np.array(f['equipment/detector/distance'])  # microns
        self._pixel_size = np.array(f['equipment/detector/x_pixel_size'])  # microns
        self._obj_pixel_size = self.distance * self.wavelength / (self.shape[0] * self.pixel_size)

        f.close()

        # IMAGE PRE-PROCESSING ########################################################################################
        # Invert the probe positions
        if 'h' in flip_positions:
            self._position[:, :, 1] = np.max(self.position[:, :, 1]) - self.position[:, :, 1]
        if 'v' in flip_positions:
            self._position[:, :, 0] = np.max(self.position[:, :, 0]) - self.position[:, :, 0]

        # Flip the diffraction images
        if 'h' in flip_images:
            self._im_data = np.flip(self.im_data, 3)
            self._bkgd = np.flip(self._bkgd, 1)
        if 'v' in flip_images:
            self._im_data = np.flip(self.im_data, 2)
            self._bkgd = np.flip(self._bkgd, 0)

        # Perform background subtraction
        if background_subtract:
            self._im_data[:, :] = self.im_data[:, :] - self.bkgd
            self._im_data[self.im_data < 0] = 0

        # Perform vertical bleed correction
        vbleed = np.quantile(self.im_data, vbleed_correct, axis=2)
        vbleed = np.reshape(np.tile(vbleed, self.shape[-2]), self.im_data.shape)
        self._im_data = self.im_data - vbleed
        self._im_data[self.im_data < 0] = 0

        # Perform thresholding to further reduce noise
        self._im_data[self.im_data < threshold*np.max(self.im_data)] = 0

        # Squeeze rotation axis if necessary
        self._is3d = self.num_rotations > 1
        if not self._is3d:
            self._im_data = np.squeeze(self._im_data)
            self._position = np.squeeze(self.position)[:, :2]

    @property
    def is3d(self):
        return self._is3d

    @property
    def max_translation(self):
        if self.is3d:
            return np.max(np.abs(self.position[:, :, :2]), axis=0)
        else:
            return np.max(np.abs(self.position), axis=0)


class CollectData(PtychoData):
    def __init__(self, num_rotations, num_translations, title, im_shape, pixel_size, distance, energy, verbose=False):
        super().__init__()

        # General parameters
        self.verbose = verbose
        self._num_entries = num_translations * num_rotations  # Total number of takes (all translations and rotations)
        self._num_translations = num_translations
        self._num_rotations = num_rotations
        self._n = 0  # Current take
        self._i = 0  # Current rotation
        self._j = 0  # Current translation
        self._shape = im_shape  # image resolution
        self._bkgd = np.zeros(im_shape)

        # Pre-allocate memory for these arrays, as they can get quite large
        self._position = np.empty((self.num_rotations, self.num_translations, 3))
        self._im_data = np.empty((self.num_rotations, self.num_translations,
                                  self._shape[0], self._shape[1]))

        # These parameters probably won't change
        self._energy = energy  # photon energy in eV
        self._wavelength = 1.240 / energy  # wavelength in microns
        self._pixel_size = pixel_size  # pixel side length in microns
        self._distance = distance * 1e6  # sample-to-detector distance in microns

        # Parameters for saving the data afterward
        self.title = title
        self.is_finalized = False
        return

    def print(self, text):
        if self.verbose:
            print(text)

    def record_data(self, position, im_data):
        # Record the position and image data
        self._position[self._i, self._j] = position  # Record in millimeters / degrees
        self._im_data[self._i, self._j] = im_data

        # This next bit is all about making sure the measurements from a given rotation all stay together.
        # Increment j until it fills up, then reset and increment i.
        self._n += 1
        self._j += 1
        ret_code = self._j == self.num_translations
        if ret_code:
            self._j = 0
            self._i += 1
        self.print(f'Take {self._n} of {self.num_entries} recorded!')
        return ret_code

    def record_background(self, im_data):
        self._bkgd = im_data

    def finalize(self, timestamp, cropto):
        if not self.is_finalized:
            # Translate all the positions so that the minimum is zero.
            self._position = self.position - np.min(self.position, axis=1)
            print('Finalizing...')

            # print('Subtracting background...')
            # if self._bkgd is not None:
            #     self._im_data[:, :] = self._im_data[:, :] - self._bkgd
            # else:
            #     # self._im_data = self._im_data - np.median(self._im_data)
            #     # self._im_data[self._im_data < 0] = 0
            #     pass

            print('Cropping to square...')
            if np.any(np.array(self._shape) > cropto):
                # If desired, crop the image down to save space.
                d = cropto // 2
                # Be careful with this... scipy doesn't always hit the center exactly.
                cy, cx = ndimage.center_of_mass(np.sum(self._im_data, axis=(0, 1)))
                cx = int(cx)
                cy = int(cy)
                # Crop the image data down to the desired size.
                self._im_data = self._im_data[:, :, cy - d:cy + d + 1, cx - d:cx + d + 1]
                self._bkgd = self._bkgd[cy - d:cy + d + 1, cx - d:cx + d + 1]
                self._shape = (self.im_data.shape[-2], self.im_data.shape[-1])

            if timestamp:
                # Add the date to the end of the title. Optional for simulated data.
                self.title = self.title + '-' + date.today().isoformat()

            self.is_finalized = True
        return

    def show(self, ij=(0, 0)):
        i = ij[0]
        j = ij[1]
        plt.subplot(121, xticks=[], yticks=[], title='Linear')
        plt.imshow(self._im_data[i, j], cmap='plasma')
        plt.subplot(122, xticks=[], yticks=[], title='Logarithmic')
        plt.imshow(np.log(self._im_data[i, j] + 1), cmap='plasma')
        plt.tight_layout()
        plt.show()

    def save_to_pty(self, timestamp=True, cropto=512, dtype=None):
        if self._n == self.num_entries:
            self.finalize(timestamp, cropto)
        else:
            if input('Scan not complete! Are you sure you want to save? (Y/n)').lower() == 'n':
                return
            else:
                self.title = self.title + '-INCOMPLETE'

        print(f'Saving data as {self.title}.pty')

        # Create file using HDF5 protocol.
        os.chdir(os.path.dirname(__file__))
        os.chdir('..')
        f = h5py.File(f'data/{self.title}.pty', 'w')

        # Save collected data
        data = f.create_group('data')
        data.create_dataset('data', data=self._im_data, dtype=dtype)
        data.create_dataset('background', data=self._bkgd)
        data.create_dataset('position', data=self._position)

        # Save experimental parameters
        equip = f.create_group('equipment')
        source = equip.create_group('source')
        source.create_dataset('energy', data=self.energy)
        source.create_dataset('wavelength', data=self.wavelength)
        detector = equip.create_group('detector')
        detector.create_dataset('distance', data=self.distance)
        detector.create_dataset('x_pixel_size', data=self.pixel_size)
        detector.create_dataset('y_pixel_size', data=self.pixel_size)

        f.close()
        return


class GenerateData2D(CollectData):
    def __init__(self, max_shift, probe_radius, overlap, title, im_size=127, pattern='rect', flip=False, show=False):
        X, Y, N = xy_scan(pattern, max_shift, max_shift, (1-overlap)*2*probe_radius)
        max_shift *= 1e3  # Convert shift from mm to microns
        probe_radius *= 1e3  # Convert to microns
        super().__init__(1, N, title, (im_size, im_size), 5.5, 0.1, 2.0, verbose=show)

        # Generate a probe
        probe_pixel_size = self.distance * self.wavelength / (im_size * self.pixel_size)  # microns per pixel
        self._obj_pixel_size = probe_pixel_size
        probe_radius = probe_radius / probe_pixel_size  # Convert from microns to pixels
        probe_array = generate_probe(im_size, probe_radius, nophase=True)

        Xp = X * 1e3 / probe_pixel_size
        Yp = Y * 1e3 / probe_pixel_size
        Xp = Xp - np.min(Xp)
        Yp = Yp - np.min(Yp)

        # Generate an object
        obj_size = im_size + int(max_shift / probe_pixel_size)
        object_array = hf.demo_binary(obj_size)
        # The plus-one is in case the max shift has some extra decimal places, and would therefore try to sample past
        # the edge of the image.

        if show:
            # Show the probe (amplitude and phase)
            plt.figure()
            plt.subplot(121)
            plt.imshow(np.abs(probe_array), cmap='gray')
            plt.subplot(122)
            plt.imshow(np.angle(probe_array), cmap='hsv')

            # Show the object (amplitude and phase)
            plt.figure()
            plt.subplot(121)
            plt.imshow(np.abs(object_array), cmap='gray')
            plt.subplot(122)
            plt.imshow(np.angle(object_array), cmap='hsv', clim=[-np.pi, np.pi])

            # Show the overlay of all the probes on the object amplitude. Useful to check overlap.
            plt.figure()
            ax = plt.subplot(111)
            ax.imshow(np.abs(object_array))
            for i in range(self.num_entries):
                ax.add_artist(Circle((im_size/2 + Yp[i], im_size/2 + Xp[i]), probe_radius, color='r', fill=False))

        if show:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        else:
            fig, ax1, ax2, ax3 = None, None, None, None
        frames = []
        entries = range(self.num_entries)
        if not show:
            entries = progressbar.progressbar(entries)

        # This for-loop is where the actual diffraction patterns are made!!
        for i in entries:
            shifted_object = hf.shift(np.copy(object_array), (-Yp[i], -Xp[i]), crop=im_size)
            exit_wave = probe_array * shifted_object
            diffraction = np.abs(hf.fft(exit_wave)) ** 2

            if show:
                frames.append([ax1.imshow(np.abs(object_array)),
                               ax1.add_artist(
                                   Circle((im_size/2 + Yp[i], im_size/2 + Xp[i]),
                                          probe_radius, color='r', fill=False)),
                               ax2.imshow(np.abs(exit_wave)),
                               ax3.imshow(diffraction)])
            position = np.array([Y[i], X[i], 0])
            if flip:
                diffraction = np.flipud(diffraction)
            self.record_data(position, diffraction)

        if show:
            # Animate the set of all the diffraction patterns
            vid = ArtistAnimation(fig, frames, interval=800, repeat_delay=0)

            # Plot diffraction before detection
            plt.figure()
            ax1 = plt.subplot(121)
            plt.imshow(np.log(self._im_data[0, 0]))

        # Simulate detection process (saturation, bitdepth, background, summed frames)
        self.detect(0.75, 24, 50, 10)

        if show:
            # Plot diffraction after detection
            plt.subplot(122, sharex=ax1, sharey=ax1)
            plt.imshow(np.log(self._im_data[0, 0]))
            plt.show()
            print(self)

        # Save the whole data to a pty file
        self.save_to_pty(timestamp=False)

    def detect(self, saturation=1.0, bitdepth=16, bkgd=0, frames=1, seed=None):
        """
        Simulate the process of detection on a set of ptychography data

        :param saturation: Fraction of the dynamic range reached by the brightest pixel
        :param bitdepth: Number of bits given to each pixel.
        :param bkgd: Average number of counts per pixel not attributed to the measured signal (dark signal)
        :param frames: Number of frames summed for each "take"
        :param seed: random seed to use for noise
        """
        signal_max = np.max(self._im_data)
        pixel_max = 2 ** bitdepth - 1
        if seed is None:
            seed = np.random.randint(2 ** 31)
        rng = np.random.default_rng(seed)

        # Apply saturation
        self.print(f'\tSaturation: {int(100 * saturation)}%')
        self._im_data = np.clip(self._im_data * saturation, None, signal_max)

        if bitdepth > 0:
            # Scale data to bit depth:
            self.print(f'\tBit depth: {bitdepth}-bit')
            self._im_data = np.around(self._im_data * pixel_max / signal_max) + bkgd

            # Apply bit reduction
            self._im_data = self._im_data.astype('int')

            # Apply noise and sum frames
            self.print(f'\tNoise model: Poisson\n'
                       f'\tNoise seed: {seed}\n'
                       f'\tFrames summed in each measurement: {frames}')
            im_array = [np.clip(rng.poisson(self._im_data), None, pixel_max) for x in range(frames)]
            self._im_data = np.sum(im_array, axis=0)


def generate_probe(im_size, probe_radius, nophase=False):
    probe_array = np.zeros((im_size, im_size))
    probe_center = im_size / 2

    # Generate the amplitude (blurred disk)
    probe_array[draw.disk((probe_center, probe_center), probe_radius)] = 1
    probe_array = ndimage.gaussian_filter(probe_array, 2)

    if nophase:
        return probe_array + 0j

    # Generate the phase (2D quadratic to represent a diverging/converging beam)
    X, Y = np.meshgrid(np.arange(im_size), np.arange(im_size))
    probe_phase = ((X - probe_center) / 20) ** 2 + ((Y - probe_center) / 20) ** 2

    # Generate the probe as the combined amplitude and phase
    return probe_array * np.exp(1j * probe_phase)


if __name__ == '__main__':
    q = 0.35
    t0 = 0.0
    t1 = 2e-4
    t2 = 5e-4
    data0 = LoadData('test-2021-07-20.pty', vbleed_correct=q, threshold=t0)
    data1 = LoadData('test-2021-07-20.pty', vbleed_correct=q, threshold=t1)
    data2 = LoadData('test-2021-07-20.pty', vbleed_correct=q, threshold=t2)

    i = 23
    plt.figure(tight_layout=True)
    plt.subplot(131, xticks=[], yticks=[], title='No thresholding')
    plt.imshow(np.log(data0.im_data[i]+1), cmap='gray')
    plt.subplot(132, xticks=[], yticks=[], title=f'Threshold = {t1}')
    plt.imshow(np.log(data1.im_data[i]+1), cmap='gray')
    plt.subplot(133, xticks=[], yticks=[], title=f'Threshold = {t2}')
    plt.imshow(np.log(data2.im_data[i]+1), cmap='gray')
    plt.show()
