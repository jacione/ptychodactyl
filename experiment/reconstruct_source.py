import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import draw
from matplotlib.patches import Rectangle
from matplotlib.animation import ArtistAnimation
import progressbar
from abc import ABC, abstractmethod
from experiment.helper_funcs import ifft, random, shift
from experiment.ptycho_data import LoadData


def Reconstruction(data: LoadData):
    if data.is3d:
        return Recon3D(data)
    else:
        return Recon2D(data)


class Recon(ABC):
    @abstractmethod
    def __init__(self, data: LoadData):
        self.data = data
        print(self.data)
        self.probe = init_probe(self.data.im_data)
        # self.probe = init_probe_flat(self.data.shape)
        self.crop = self.data.shape[0]
        self.pixel_size = data.distance * data.wavelength / (data.shape[0] * data.pixel_size)
        max_translation = np.around(self.data.max_translation / self.pixel_size).astype('int')
        self.object = init_object(data.shape + max_translation)
        # print(self.object.shape)
        self.temp_object = np.zeros_like(self.object)
        self.rng = np.random.default_rng()
        self.probe_energy = np.max(np.sum(self.data.im_data, axis=(1, 2)))
        self.ix, self.iy = np.indices(self.probe.shape)
        self.__algs = {}

    @abstractmethod
    def run(self, num_iterations, algorithm, animate):
        pass

    @abstractmethod
    def show_object_and_probe(self):
        pass

    @abstractmethod
    def __apply_update(self, x, y, object_update, probe_update):
        pass

    @abstractmethod
    def __correct_probe(self):
        pass


class Recon3D(Recon):
    def __init__(self, data: LoadData):
        super().__init__(data)

        self.__algs = {}
        pass

    def run(self, num_iterations, algorithm, animate=False):
        pass

    def show_object_and_probe(self):
        pass

    def __apply_update(self, x, y, object_update, probe_update):
        pass

    def __correct_probe(self):
        pass


class Recon2D(Recon):
    def __init__(self, data: LoadData):
        super().__init__(data)
        assert not self.data.is3d, "Cannot reconstruct 3D data with 2D reconstruction algorithms"

        # This is the translation data IN UNITS OF PIXELS
        self.translation = self.data.position / self.pixel_size
        self.__correct_probe()
        self.shifted_im_data = np.sqrt(np.fft.ifftshift(self.data.im_data, axes=(1, 2)))

        self.__algs = {
            'epie': self.__epie,
            'rpie': self.__rpie
        }

    def run(self, num_iterations, algorithm, apply_mask_at=(), animate=True):
        try:
            update_function = self.__algs[algorithm]
        except KeyError:
            print(f"Warning: '{algorithm}' is not a recognized reconstruction algorithm. Skipping...")
            return
        fig, ax1, ax2, ax3, ax4 = setup_figure()
        frames = []
        for x in progressbar.progressbar(range(num_iterations)):
            entries = np.arange(self.data.num_entries)
            self.rng.shuffle(entries)
            update_param = x / num_iterations
            for i in entries:
                update_probe = x > 0
                update_function(i, update_param, update_probe)
            if x in apply_mask_at:
                self.object = self.object * self.__probe_mask()
            if animate:
                frames.append([ax1.imshow(np.abs(self.object), cmap='bone'),
                               ax2.imshow(np.angle(self.object), cmap='hsv'),
                               ax3.imshow(np.abs(self.probe), cmap='bone'),
                               ax4.imshow(np.angle(self.probe), cmap='hsv')])
        if animate:
            vid = ArtistAnimation(fig, frames, interval=100, repeat_delay=0)
        self.show_object_and_probe()

    def __apply_update(self, x, y, object_update, probe_update=None):
        self.object[x + self.ix, y + self.iy] = self.object[x + self.ix, y + self.iy] + object_update
        if probe_update is not None:
            self.probe = self.probe + probe_update
            self.__correct_probe()

    def __pre_pie(self, i):
        # Take a slice of the object that's the same size as the probe
        x = int(self.translation[i, 0] + 0.5)
        y = int(self.translation[i, 1] + 0.5)
        region = self.object[x + self.ix, y + self.iy]
        # region = hf.shift(self.object, self.translation[i], crop=self.data.shape[0], subpixel=False)
        psi1 = self.probe * region
        PSI1 = np.fft.fft2(psi1)
        PSI2 = self.shifted_im_data[i] * PSI1 / np.abs(PSI1)
        psi2 = np.fft.ifft2(PSI2)
        d_psi = psi2 - psi1
        return x, y, region, d_psi

    def __epie(self, i, param, update_probe=True):
        x, y, region, d_psi = self.__pre_pie(i)
        alpha = np.sqrt(1.25-param)
        beta = 1-param
        object_update = alpha * d_psi * np.conj(self.probe) / np.max(np.abs(self.probe)) ** 2
        probe_update = None
        if update_probe:
            probe_update = beta * d_psi * np.conj(region) / np.max(np.abs(region)) ** 2
        self.__apply_update(x, y, object_update, probe_update)

    def __rpie(self, i, param, update_probe=True):
        x, y, region, d_psi = self.__pre_pie(i)
        alpha = 0.05 + 0.45 * param ** 2
        beta = 0.25 + 0.75 * param
        object_update = d_psi * np.conj(self.probe) / \
                        (alpha*np.max(np.abs(self.probe))**2 + (1-alpha)*np.abs(self.probe)**2)
        probe_update = None
        if update_probe:
            probe_update = d_psi * np.conj(region) / \
                           (beta*np.max(np.abs(region))**2 + (1-beta)*np.abs(region)**2)
        self.__apply_update(x, y, object_update, probe_update)

    def __correct_probe(self):
        self.probe = self.probe * np.sqrt(self.probe_energy / np.sum(np.abs(self.probe) ** 2)) / self.data.shape[0]
        self.probe = center(self.probe)

    def __probe_mask(self):
        probe_mask = np.zeros(self.probe.shape)
        probe_mask[np.abs(self.probe) > np.abs(np.max(self.probe)) / 4] = 1
        object_mask = np.zeros(self.object.shape)
        for t in self.translation:
            x = int(t[0] + 0.5)
            y = int(t[1] + 0.5)
            object_mask[x+self.ix, y+self.ix] += probe_mask
        object_mask = object_mask + 0.1
        object_mask[object_mask > 0.5] = 1
        object_mask = ndimage.gaussian_filter(object_mask, 2)
        return object_mask

    def show_probe_positions(self):
        coords = -self.translation + self.probe.shape[0] / 2
        x, y = coords[:, 1], coords[:, 0]
        plt.scatter(x, y)
        plt.xlim(0, self.object.shape[0])
        plt.ylim(0, self.object.shape[1])
        plt.show()

    def show_object_and_probe(self, i=None):
        fig, ax1, ax2, ax3, ax4 = setup_figure()
        ax1.imshow(np.abs(self.object), cmap='bone')
        if i is not None:
            ax1.add_artist(Rectangle(-np.flip(self.translation[i]), self.data.shape[0], self.data.shape[1],
                                     fill=False, color='r'))
        ax2.imshow(np.angle(self.object), cmap='hsv')
        ax3.imshow(np.abs(self.probe), cmap='bone')
        ax4.imshow(np.angle(self.probe), cmap='hsv')
        plt.show()


def center(image):
    ctr = np.asarray(image.shape) / 2 - 0.5
    c_mass = np.array(ndimage.center_of_mass(np.abs(image)))
    shift_amt = ctr - c_mass
    ret_image = shift(image, shift_amt, subpixel=True)
    return ret_image


def init_object(shape):
    object_array = np.zeros(shape) + 0.1 + 0j
    object_array = object_array * np.exp(1j*random(object_array.shape, 0, 2*np.pi))
    return object_array


def init_probe(data):
    diff_array = np.mean(data, axis=0)
    probe_array = ifft(np.sqrt(diff_array))
    probe_array = ndimage.gaussian_filter(np.abs(probe_array), 3) * np.exp(1j*np.angle(probe_array))
    ax = plt.subplot(121, title='Sum of diffraction patterns')
    plt.imshow(diff_array)
    plt.subplot(122, sharex=ax, sharey=ax, title='Initial probe guess')
    plt.imshow(np.abs(probe_array))
    plt.show()
    return probe_array


def init_probe_flat(shape):
    probe_array = np.zeros(shape)
    probe_center = shape[0] / 2 - 0.5
    probe_radius = shape[0] / 4
    rr, cc = draw.disk((probe_center, probe_center), probe_radius)
    probe_array[rr, cc] = 1
    probe_array = probe_array * np.exp(1j*random(probe_array.shape, 0, 2*np.pi))
    return probe_array


def setup_figure():
    kw = {'xticks': [], 'yticks': []}
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='row', sharey='row', tight_layout=True, subplot_kw=kw)
    ax1.set_title('Amplitude')
    ax2.set_title('Phase')
    ax1.set_ylabel('Object')
    ax3.set_ylabel('Probe')
    return fig, ax1, ax2, ax3, ax4


if __name__ == '__main__':
    pass
