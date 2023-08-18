"""
Classes for ptychographic image reconstruction.

Currently implemented:
    - `Extended Ptychographic Iterative Engine (ePIE) <https://doi.org/10.1016/j.ultramic.2009.05.012>`_, 2D
    - `Regularized Ptychographic Iterative Engine (rPIE) <https://doi.org/10.1364/OPTICA.4.000736>`_, 2D
Planned:
    - `Coupled ptycho-tomography (CPT) <https://doi.org/10.1364/OL.42.003169>`_, 3D
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import numpy as np
import tqdm
import yaml
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy import ndimage as ndi
from skimage import draw
from skimage.restoration import unwrap_phase

from ptycho.ptycho_data import LoadData
import ptycho.utils as ut
from ptycho.plotting import comp_to_rgb


class Dataset:
    def __init__(self, filepath, prep_config):
        """
        :param filepath: pty datafile. Must be structured by ptychodactyl
        :param prep_config: dictionary with preprocessing configuration
        """

        self._config = prep_config
        det = self._config["det"]
        with h5py.File(filepath, 'r') as f:
            print(f"Loading data for {det}...")
            self._data = np.array([f[f"data/{key}/{det}"] for key in f["data"].keys()])
            self._position = np.array([f[f"data/{key}/position"] for key in f["data"].keys()])
            if det not in f['equipment'].keys():
                det = det.replace('det', 'detector')
            self._bkgd = np.array(f[f"equipment/{det}/background"])
            self._det_specs = {k: v for k, v in f[f"equipment/{det}"].attrs.items()}
            self._det_specs["energy"] = f["equipment/source"].attrs["energy"]
            self._det_specs["wavelength"] = f["equipment/source"].attrs["wavelength"]

        # FIXME DELETE THE FOLLOWING BLOCK AFTER FIXING experiment.Experiment DATA STORAGE
        self._position = self._position[:, 1:, :] * 1e3
        self._position -= np.min(self._position, axis=1)
        self._det_specs["distance"] *= 1e6

        self._n_rot, self._n_trn, self._x_pix, self._y_pix = self._data.shape
        self._data = np.reshape(self._data, (-1, self._x_pix, self._y_pix))

        # Apply all the preprocessing
        self.subtract_background()
        self.center_crop_bin()
        self.vbleed_correct()
        self.flip_images()

        # If there is no rotational data, then the images can just be stacked along one axis.
        if self.is3d:
            self._data = np.reshape(self._data, (self._n_rot, self._n_trn, self._x_pix, self._y_pix))
        else:
            self._position = np.squeeze(self._position[:, :, :2])

        self.prepare_to_phase()
        return

    def __getitem__(self, item):
        return self._det_specs[item]

    @property
    def data(self):
        return self._data

    @property
    def position(self):
        return self._position

    @property
    def is3d(self):
        return self._n_rot > 1

    @property
    def shape(self):
        return np.array([self._x_pix, self._y_pix])

    @property
    def n_images(self):
        return self._n_rot * self._n_trn

    @property
    def max_translation(self):
        if self.is3d:
            return np.max(np.abs(self._position[:, :, :2]), axis=0)
        else:
            return np.max(np.abs(self._position), axis=0)

    @property
    def obj_pixel_size(self):
        return self["distance"] * self["wavelength"] / (self._x_pix * self["x_pixel_size"])

    @property
    def obj_image_shape(self):
        return self.shape + np.ceil(self.max_translation / self.obj_pixel_size).astype('int')

    @property
    def norm(self):
        return np.max(np.sum(self._data, axis=(-2, -1)))

    def subtract_background(self):
        print("Subtracting background...")
        self._data = np.clip(self._data - self._bkgd, 0, None)

    def vbleed_correct(self):
        """
        Custom filter that attempts to correct for smearing in readout columns
        Background reading obaut smearing: https://www.photometrics.com/learn/imaging-topics/saturation-and-blooming
        """
        print("Correcting vertical smear...")
        vbleed = np.quantile(self._data, self._config["vbleed_correct"], axis=1)
        vbleed = np.reshape(np.tile(vbleed, self._x_pix), self._data.shape)
        self._data = np.clip(self._data - vbleed, 0, None)

    def center_crop_bin(self):
        if self._config["center"]:
            print("Centering images...")
            im_sum = np.sum(self._data, axis=0)
            ctr = np.asarray(im_sum.shape) / 2 - 0.5
            im_sum[im_sum < np.quantile(im_sum, 0.8)] = 0
            c_mass = np.array(ndi.center_of_mass(im_sum))
            shift_x, shift_y = ctr - c_mass
            self._data = np.roll(self._data, (0, int(shift_x+0.5), int(shift_y+0.5)))
        if self._config["crop"]:
            print("Cropping images...")
            _, x, y = self._data.shape
            cx = (x - self._config["crop"]) // 2
            cy = (y - self._config["crop"]) // 2
            self._data = self._data[:, cx:x-cx, cy:y-cy]
            _, self._x_pix, self._y_pix = self._data.shape
        if self._config["binning"]:
            print("Binning images...")
            b = self._config["binning"]
            kernel = np.zeros((1, 2*b-1, 2*b-1))
            kernel[:, :b, :b] = 1
            self._data = ndi.convolve(self._data, kernel, mode="constant")[:, ::b, ::b]
            _, self._x_pix, self._y_pix = self._data.shape
            self._det_specs["x_pixel_size"] *= b
            self._det_specs["y_pixel_size"] *= b

        print(f"Final shape: {self._data.shape}")

    def threshold(self):
        # TODO: apply thresholding, maybe based on quantile?
        pass

    def flip_images(self):
        """Flips images along the vertical, horizontal, and/or diagonal axes"""
        v, h, x = self._config["flip"]
        if True in (v, h, x):
            print("Flipping images...")
        if v:
            self._data = np.flip(self._data, axis=1)
            self._bkgd = np.flip(self._bkgd, axis=0)
        if h:
            self._data = np.flip(self._data, axis=2)
            self._bkgd = np.flip(self._bkgd, axis=1)
        if x:
            self._data = np.swapaxes(self._data, 1, 2)
            self._bkgd = np.swapaxes(self._bkgd, 0, 1)

    def prepare_to_phase(self):
        print("Preparing for phasing...")
        self._data = np.sqrt(np.fft.ifftshift(self._data, axes=(1, 2)))


class Recon(ABC):
    @abstractmethod
    def __init__(self, config):
        """
        Abstract parent class for ptychographic reconstructions.
        """
        self._config = config
        self._file = f"{config['data_dir']}/{config['title']}"
        if not self._file.endswith(".pty"):
            self._file += ".pty"
        self._file = Path(self._file)
        if not self._file.exists():
            raise FileNotFoundError(f"File {self._file} does not exist.")

        self.diff_config = config["diffraction"]
        self.diffraction = Dataset(self._file, self.diff_config)

        # self.probe_config = config["probe"]
        # self.probe_data = Dataset(self._file, self.probe_config)
        print(f"Initializing object array...")
        self.object = init_object(self.diffraction.obj_image_shape)
        print("Initializing probe array...")
        self.probe = init_probe(self.diffraction.data)
        self.probe_energy = self.diffraction.norm
        self.temp_object = np.zeros_like(self.object)
        self.rng = np.random.default_rng()
        self.rows, self.cols = np.indices(self.diffraction.shape)
        self._algs = {}

    @abstractmethod
    def run(self):
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

    def save_reconstruction(self):
        """Save the reconstruction to the original PTY file"""
        f = h5py.File(self._file, 'r+')
        try:
            group = f.create_group('reconstruction')
        except ValueError:
            del f['reconstruction']
            group = f.create_group('reconstruction')
        group.create_dataset('object_amplitude', data=np.abs(self.object))
        group.create_dataset('object_phase', data=np.angle(self.object))
        group.create_dataset('probe_amplitude', data=np.abs(self.probe))
        group.create_dataset('probe_phase', data=np.angle(self.probe))

        f.close()


class Recon2D(Recon):
    def __init__(self, config):
        """
        Create a 2D ptychographic reconstructor

        :param config: configuration dictionary
        :type config: dict
        """
        super().__init__(config)
        assert not self.diffraction.is3d, "Cannot reconstruct 3D data with 2D reconstruction algorithms"

        # This is the translation data IN UNITS OF PIXELS
        self.translation = self.diffraction.position / self.diffraction.obj_pixel_size
        self._correct_probe()

        # Dictionary of all currently supported algorithms
        self._algs = {
            'epie': lambda i, o, p, u: self._epie(i, o, p, u),
            'rpie': lambda i, o, p, u: self._rpie(i, o, p, u)
        }

    def run(self):
        """
        Perform a reconstruction using the provided parameters.
        """

        # Define what the update strength will be at each iteration based on the initial and final values provided

        # Make sure the provided algorithm is one it knows
        all_recipes = [val for key, val in self._config.items() if key.startswith("recipe")]
        for recipe in all_recipes:
            try:
                update_function = self._algs[recipe["algorithm"]]
            except KeyError:
                print(f"Warning: '{recipe['algorithm']}' is not a recognized reconstruction algorithm.")
                return

            # Set up the animation (returns all Nones if animate is False)
            fig, ax1, ax2, ax3, ax4 = setup_figure(recipe['animate'])
            frames = []

            entries = np.arange(self.diffraction.n_images)
            o_param = np.linspace(recipe["obj_update"][0], recipe["obj_update"][1], recipe["iterations"])
            p_param = np.linspace(recipe["prb_update"][0], recipe["prb_update"][1], recipe["iterations"])
            # Perform the reconstruction
            for j in tqdm.tqdm(range(recipe['iterations'])):
                self.rng.shuffle(entries)  # Shuffling the order with each iteration helps the reconstruction
                for i in entries:
                    update_function(i, o_param[j], p_param[j], j > 0)
                self._correct_phase()

                if recipe['animate']:
                    frames.append([ax1.imshow(np.abs(self.object), cmap='gray'),
                                   ax2.imshow(np.angle(self.object), cmap='hsv'),
                                   ax3.imshow(np.abs(self.probe), cmap='gray'),
                                   ax4.imshow(np.angle(self.probe), cmap='hsv')])
            if recipe['animate']:
                vid = ArtistAnimation(fig, frames, interval=500, repeat_delay=0)
                plt.show()

    def _dampen_phase(self):
        self.object = np.abs(self.object) * np.exp(1j*ndi.gaussian_filter(np.angle(self.object), 1))
        self.probe = np.abs(self.probe) * np.exp(1j*ndi.gaussian_filter(np.angle(self.probe), 1))

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
        PSI2 = self.diffraction.data[i] * PSI1 / np.abs(PSI1)
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
        self.probe = self.probe * np.sqrt(self.probe_energy / np.sum(np.abs(self.probe) ** 2)) / self.probe.shape[0]

        # Step 3: Shift the probe so that it is centered in the array
        self.probe = ut.center_array(self.probe)

        # Step 4: Apply a Gaussian filter on the probe phase to dampen really high frequency noise
        # self.probe = np.abs(self.probe) * np.exp(1j*ndi.gaussian_filter(np.angle(self.probe), 0.25))

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
        object_mask = ndi.gaussian_filter(object_mask, 2)
        return object_mask

    def show_probe_positions(self):
        """Show a scatter plot of the probe positions in the scan."""
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
        obj_length = self.diffraction.obj_pixel_size * self.object.shape[0]
        pro_length = self.diffraction.obj_pixel_size * self.probe.shape[0]
        plt.subplot(121, title='Object', xlabel='[microns]', ylabel='[microns]')
        plt.imshow(comp_to_rgb(self.object), extent=(0, obj_length, 0, obj_length))
        plt.subplot(122, title='Probe', xlabel='[microns]', ylabel='[microns]')
        plt.imshow(comp_to_rgb(self.probe), extent=(0, pro_length, 0, pro_length))
        plt.show()


def init_object(shape):
    """
    Initialize an object to have everywhere a small nonzero amplitude and a random phase.

    :param shape: Shape of the object array
    :type shape: (int, int)
    :return: Initialized object array
    :rtype: np.ndarray
    """
    object_array = np.zeros(shape) + 0.01 + 0j
    object_array = object_array * np.exp(1j*ut.random(object_array.shape, 0, 2*np.pi))
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
    probe_array = ut.ifft(diff_array)
    probe_array = ndi.gaussian_filter(np.abs(probe_array), sigma) * np.exp(1j*np.angle(probe_array))
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
    probe_array = probe_array * np.exp(1j*ut.random(probe_array.shape, 0, 2*np.pi))
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
    conf_file = Path(__file__).parents[1] / "config_example/reconstruction_config.yml"
    conf = yaml.safe_load(conf_file.read_text())
    rec = Recon2D(conf)
    rec.run()
    rec.save_reconstruction()
    rec.show_object_and_probe()
