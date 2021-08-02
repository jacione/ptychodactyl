"""
Classes for ptychographic image reconstruction.

Currently implemented:
    - 2D ePIE (https://doi.org/10.1016/j.ultramic.2009.05.012)
    - 2D rPIE (https://doi.org/10.1364/OPTICA.4.000736)
Planned:
    - 3D CPT (https://doi.org/10.1364/OL.42.003169)
"""

import h5py
import os
from abc import ABC, abstractmethod

import numpy as np
import progressbar
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy import ndimage
from skimage import draw
from skimage.restoration import unwrap_phase

from ptycho_data import LoadData
from utils.general import ifft, random, shift
from utils.plotting import comp_to_rgb


def Reconstruction(filename, **specs):
    """
    Load a PTY file and construct the appropriate Recon subclass.

    :param filename: name of the datafile
    :type filename: str
    :param specs: keyword arguments to pass to LoadData
    :return: Recon2D or Recon3D instance
    :rtype: Recon
    """
    data = LoadData(filename, **specs)
    if data.is3d:
        return Recon3D(data)
    else:
        return Recon2D(data)


class Recon(ABC):
    @abstractmethod
    def __init__(self, data: LoadData):
        """
        Abstract parent class for ptychographic reconstructions.

        :param data: Data set that will be reconstructed
        :type data: LoadData
        """
        self.data = data
        print(self.data)
        self.probe = init_probe(self.data.im_data)
        # self.probe = init_probe_flat(self.data.shape)
        self.crop = self.data.shape[0]
        self.pixel_size = self.data.obj_pixel_size
        max_translation = np.around(self.data.max_translation / self.pixel_size).astype('int')
        self.object = init_object(data.shape + max_translation)
        # print(self.object.shape)
        self.temp_object = np.zeros_like(self.object)
        self.rng = np.random.default_rng()
        self.probe_energy = np.max(np.sum(self.data.im_data, axis=(1, 2)))
        self.rows, self.cols = np.indices(self.probe.shape)
        self._algs = {}

    @abstractmethod
    def run(self, run_specs, animate):
        pass

    @abstractmethod
    def show_object_and_probe(self):
        pass

    @abstractmethod
    def _apply_update(self, x, y, object_update, probe_update):
        pass

    @abstractmethod
    def _correct_probe(self):
        pass

    def save(self):
        """Save the reconstruction to the original PTY file"""
        os.chdir(os.path.dirname(__file__))
        os.chdir('data')
        f = h5py.File(self.data.file, 'r+')
        try:
            group = f.create_group('reconstruction')
            group.create_dataset('object_amplitude', data=np.abs(self.object))
            group.create_dataset('object_phase', data=np.angle(self.object))
            group.create_dataset('probe_amplitude', data=np.abs(self.probe))
            group.create_dataset('probe_phase', data=np.angle(self.probe))
        except ValueError:
            group = f['reconstruction']
            group['object_amplitude'].write_direct(np.abs(self.object))
            group['object_phase'].write_direct(np.angle(self.object))
            group['probe_amplitude'].write_direct(np.abs(self.probe))
            group['probe_phase'].write_direct(np.angle(self.probe))

        f.close()


class Recon2D(Recon):
    def __init__(self, data):
        """
        Create a 2D ptychographic reconstructor

        :param data: Data set that will be reconstructed
        :type data: LoadData
        """
        super().__init__(data)
        assert not self.data.is3d, "Cannot reconstruct 3D data with 2D reconstruction algorithms"

        # This is the translation data IN UNITS OF PIXELS
        self.translation = self.data.position / self.pixel_size
        self._correct_probe()
        self.shifted_im_data = np.sqrt(np.fft.ifftshift(self.data.im_data, axes=(1, 2)))

        # Dictionary of all currently supported algorithms
        self._algs = {
            'epie': lambda i, o, p, u: self._epie(i, o, p, u),
            'rpie': lambda i, o, p, u: self._rpie(i, o, p, u)
        }

    def run(self, run_specs, animate=False):
        """
        Perform a reconstruction using the provided parameters.

        :param run_specs: parameters for this reconstruction
        :type run_specs: dict
        :param animate: if True, saves a frame after each iteration and animates them all at the end. Default is False.
        :type animate: bool
        """

        # Load parameters from run_specs
        algorithm = run_specs['algorithm']
        num_iterations = run_specs['num_iterations']
        o_i = run_specs['obj_up_initial']
        o_f = run_specs['obj_up_final']
        p_i = run_specs['pro_up_initial']
        p_f = run_specs['pro_up_final']

        # Define what the update strength will be at each iteration based on the initial and final values provided
        o_param = np.linspace(o_i, o_f, num_iterations)
        p_param = np.linspace(p_i, p_f, num_iterations)

        # Make sure the provided algorithm is one it knows
        try:
            update_function = self._algs[algorithm]
        except KeyError:
            print(f"Warning: '{algorithm}' is not a recognized reconstruction algorithm.")
            return

        # Set up the animation (returns all Nones if animate is False
        fig, ax1, ax2, ax3, ax4 = setup_figure(animate)
        frames = []

        # Perform the reconstruction
        entries = np.arange(self.data.num_entries)
        for j in progressbar.progressbar(range(num_iterations)):
            self.rng.shuffle(entries)  # Shuffling the order with each iteration helps the reconstruction
            for i in entries:
                update_function(i, o_param[j], p_param[j], j > 0)

            if animate:
                frames.append([ax1.imshow(np.abs(self.object), cmap='bone'),
                               ax2.imshow(np.angle(self.object), cmap='hsv'),
                               ax3.imshow(np.abs(self.probe), cmap='bone'),
                               ax4.imshow(np.angle(self.probe), cmap='hsv')])
        if animate:
            vid = ArtistAnimation(fig, frames, interval=250, repeat_delay=0)

    def _apply_update(self, x, y, object_update, probe_update=None):
        """
        Update the object and probe.

        :param x: horizontal probe position (in pixels)
        :type x: int
        :param y: vertical probe position (in pixels)
        :type y: int
        :param object_update: 2D complex array
        :type object_update: np.ndarray
        :param probe_update: 2D complex array, optional
        :type probe_update: np.ndarray or None
        """
        self.object[y + self.rows, x + self.cols] = self.object[y + self.rows, x + self.cols] + object_update
        if probe_update is not None:
            self.probe = self.probe + probe_update
            self._correct_probe()

    def _pre_pie(self, i):
        """
        Performs the initial Fourier constraints common to all PIE algorithms

        :param i: image data index to use
        :type i: int
        :return: horizontal position, vertical position, original object region, partially corrected exit wave
        :rtype: (int, int, np.ndarray, np.ndarray)
        """
        # Take a slice of the object that's the same size as the probe
        y = int(self.translation[i, 0] + 0.5)
        x = int(self.translation[i, 1] + 0.5)
        region = self.object[y + self.rows, x + self.cols]
        psi1 = self.probe * region
        PSI1 = np.fft.fft2(psi1)
        PSI2 = self.shifted_im_data[i] * PSI1 / np.abs(PSI1)
        psi2 = np.fft.ifft2(PSI2)
        d_psi = psi2 - psi1
        return x, y, region, d_psi

    def _epie(self, i, o_param, p_param, update_probe):
        """
        Perform an ePIE update at a given position. The extended PIE update function is described here:
        https://doi.org/10.1016/j.ultramic.2009.05.012

        :param i: position index
        :type i: int
        :param o_param: object update strength
        :type o_param: float
        :param p_param: probe update strength
        :type p_param: float
        :param update_probe: if False, only update the object.
        :type update_probe: bool
        """
        x, y, region, d_psi = self._pre_pie(i)
        alpha = o_param
        beta = p_param
        object_update = alpha * d_psi * np.conj(self.probe) / np.max(np.abs(self.probe)) ** 2
        probe_update = None
        if update_probe:
            probe_update = beta * d_psi * np.conj(region) / np.max(np.abs(region)) ** 2
        self._apply_update(x, y, object_update, probe_update)

    def _rpie(self, i, o_param, p_param, update_probe):
        """
        Perform an rPIE update at a given position. The regularized PIE update function is described here:
        https://doi.org/10.1364/OPTICA.4.000736

        :param i: position index
        :type i: int
        :param o_param: object update strength
        :type o_param: float
        :param p_param: probe update strength
        :type p_param: float
        :param update_probe: if False, only update the object.
        :type update_probe: bool
        """
        x, y, region, d_psi = self._pre_pie(i)
        alpha = (1 - o_param) ** 2
        beta = (1 - p_param) ** 2
        object_update = d_psi * np.conj(self.probe) / \
                        (alpha*np.max(np.abs(self.probe))**2 + (1-alpha)*np.abs(self.probe)**2)
        probe_update = None
        if update_probe:
            probe_update = d_psi * np.conj(region) / \
                           (beta*np.max(np.abs(region))**2 + (1-beta)*np.abs(region)**2)
        self._apply_update(x, y, object_update, probe_update)

    def _correct_probe(self):
        """Correct the amplitude and position of the probe to prevent runaway."""

        # Step 1: Clip the brightest pixel to the same value as the second brightest
        self.probe = np.clip(np.abs(self.probe), 0, np.sort(np.abs(self.probe), axis=None)[-2]) * \
                     np.exp(1j * np.angle(self.probe))

        # Step 2: Re-normalize the total probe intensity to the brightest diffraction pattern
        self.probe = self.probe * np.sqrt(self.probe_energy / np.sum(np.abs(self.probe) ** 2)) / self.data.shape[0]

        # Step 3: Shift the probe so that it is centered in the array
        self.probe = center(self.probe)

    def _correct_phase(self):
        """
        Manually detect and remove a phase ramp from the probe and object.

        This works fairly well on simulated data, but causes problems when it's applied to real data. Until I find a
        way to fix this, I'm leaving it out of the actual reconstruction process.
        FIXME
        """
        phase = unwrap_phase(np.angle(self.probe))
        cond = np.abs(self.probe) > np.max(np.abs(self.probe))/2
        r_grad, c_grad = np.gradient(phase)
        r_grad = np.mean(r_grad[cond])
        c_grad = np.mean(c_grad[cond])
        probe_ramp = r_grad*self.rows + c_grad*self.cols

        obj_rows, obj_cols = np.indices(self.object.shape)
        obj_ramp = r_grad*obj_rows + c_grad*obj_cols

        self.probe = self.probe * np.exp(-1j*probe_ramp)
        self.object = self.object * np.exp(1j*obj_ramp)

    def _probe_mask(self):
        """
        Mask the object outside outside of the total probed area.

        This is kind of a brute-force method, and should probably be avoided. It also really gums up the performance.
        FIXME
        """
        probe_mask = np.zeros(self.probe.shape)
        probe_mask[np.abs(self.probe) > np.abs(np.max(self.probe)) / 4] = 1
        object_mask = np.zeros(self.object.shape)
        for t in self.translation:
            y = int(t[0] + 0.5)
            x = int(t[1] + 0.5)
            object_mask[y + self.rows, x + self.cols] += probe_mask
        object_mask = object_mask + 0.1
        object_mask[object_mask > 0.5] = 1
        object_mask = ndimage.gaussian_filter(object_mask, 2)
        return object_mask

    def show_probe_positions(self):
        """Show a scatter plot of all of the probe positions in the scan."""
        coords = -self.translation + self.probe.shape[0] / 2
        x, y = coords[:, 1], coords[:, 0]
        plt.imshow(self.object)
        plt.scatter(x, y, c='r')
        plt.show()

    def show_object_and_probe(self):
        """Show the object and probe in their current state of reconstruction."""
        fig, ax1, ax2, ax3, ax4 = setup_figure()
        ax1.imshow(np.abs(self.object), cmap='bone')
        ax2.imshow(np.angle(self.object), cmap='hsv')
        ax3.imshow(np.abs(self.probe), cmap='bone')
        ax4.imshow(np.angle(self.probe), cmap='hsv')

        plt.figure()
        obj_length = self.pixel_size * self.object.shape[0]
        pro_length = self.pixel_size * self.probe.shape[0]
        plt.subplot(121, title='Object', xlabel='[microns]', ylabel='[microns]')
        plt.imshow(comp_to_rgb(self.object), extent=(0, obj_length, 0, obj_length))
        plt.subplot(122, title='Probe', xlabel='[microns]', ylabel='[microns]')
        plt.imshow(comp_to_rgb(self.probe), extent=(0, pro_length, 0, pro_length))
        plt.show()


class Recon3D(Recon):
    """
    .. note::
        Not yet implemented.
    """
    def __init__(self, data):
        # """
        # Create a 3D ptychographic reconstructor
        #
        # :param data: Data set that will be reconstructed
        # :type data: LoadData
        # """
        # super().__init__(data)
        #
        # self._algs = {}
        pass

    def run(self, num_iterations, algorithm, animate=False):
        pass

    def show_object_and_probe(self):
        pass

    def _apply_update(self, x, y, object_update, probe_update):
        pass

    def _correct_probe(self):
        pass


def center(image):
    """
    Shift an image so that the center of brightness is in the center of the image. This is intended for use with a
    probe array to help keep the reconstruction centered in the frame

    :param image: Any 2D image
    :type image: np.ndarray
    :return: Centered image
    :rtype: np.ndarray
    """

    # Gets the cell-center rather than the edge-center
    ctr = np.asarray(image.shape) / 2 - 0.5

    # Take only the brightest parts of the image to get the best center
    abs_image = np.abs(image)
    abs_image[abs_image < np.quantile(abs_image, 0.6)] = 0
    c_mass = np.array(ndimage.center_of_mass(abs_image))

    # Shift the image to center the center of brightness
    shift_amt = ctr - c_mass
    ret_image = shift(image, shift_amt, subpixel=True)
    return ret_image


def init_object(shape):
    """
    Initialize an object to have everywhere a small nonzero amplitude and a random phase.

    :param shape: Shape of the object array
    :type shape: (int, int)
    :return: Initialized object array
    :rtype: np.ndarray
    """
    object_array = np.zeros(shape) + 0.1 + 0j
    object_array = object_array * np.exp(1j*random(object_array.shape, 0, 2*np.pi))
    return object_array


def init_probe(data, sigma=10):
    """
    Initialize a probe for a set of diffraction patterns. The probe is generated by taking the average of all
    diffraction patterns, inverse Fourier transforming it, and passing the resulting amplitude through a gaussian
    filter.

    :param data: Set of all diffraction patterns
    :type data: np.ndarray
    :param sigma: Width of gaussian filter in pixels
    :type sigma: float
    :return: Initialized probe array
    :rtype: np.ndarray
    """
    diff_array = np.mean(data, axis=0)
    probe_array = ifft(np.sqrt(diff_array))
    probe_array = ndimage.gaussian_filter(np.abs(probe_array), sigma) * np.exp(1j*np.angle(probe_array))
    # ax = plt.subplot(121, title='Sum of diffraction patterns')
    # plt.imshow(diff_array)
    # plt.subplot(122, sharex=ax, sharey=ax, title='Initial probe guess')
    # plt.imshow(np.abs(probe_array))
    # plt.show()
    return probe_array


def init_probe_flat(shape):
    """
    Initialize a generic probe using a top-hat function and random phase

    :param shape: Shape of the probe array
    :type shape: (int, int)
    :return: Generic probe array
    :rtype: np.ndarray
    """
    probe_array = np.zeros(shape)
    probe_center = shape[0] / 2 - 0.5
    probe_radius = shape[0] / 4
    rr, cc = draw.disk((probe_center, probe_center), probe_radius)
    probe_array[rr, cc] = 1
    probe_array = probe_array * np.exp(1j*random(probe_array.shape, 0, 2*np.pi))
    return probe_array


def setup_figure(actually_do_it=True):
    """
    Create a figure and 4 axes for plotting. This is really just a shortcut for the Recon subclasses

    :param actually_do_it: If False, returns all Nones. Default is True.
    :type actually_do_it: bool
    """
    if actually_do_it:
        kw = {'xticks': [], 'yticks': []}
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='row', sharey='row', tight_layout=True, subplot_kw=kw)
        ax1.set_title('Amplitude')
        ax2.set_title('Phase')
        ax1.set_ylabel('Object')
        ax3.set_ylabel('Probe')
        return fig, ax1, ax2, ax3, ax4
    else:
        return None, None, None, None, None


if __name__ == '__main__':
    pass
