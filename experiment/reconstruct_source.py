import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import draw
from matplotlib.patches import Rectangle
from matplotlib.animation import ArtistAnimation
import progressbar
from abc import ABC, abstractmethod
from experiment.utils.helper_funcs import ifft, random, shift
from experiment.ptycho_data import LoadData, GenerateData2D
from skimage.restoration import unwrap_phase


def parse_specs(filename):
    specs = {}
    if filename is not None:
        file = open(filename, 'r')
        for line in file.readlines():
            if "=" in line:
                key, val = map(str.strip, line.split("="))
                if 'true' in val.lower():
                    val = True
                elif 'false' in val.lower():
                    val = False
                else:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                specs[key] = val
    return specs


def Reconstruction(filename, specfile):
    specs = parse_specs(specfile)
    data = LoadData(filename, **specs)
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
        self.pixel_size = self.data.obj_pixel_size
        max_translation = np.around(self.data.max_translation / self.pixel_size).astype('int')
        self.object = init_object(data.shape + max_translation)
        # print(self.object.shape)
        self.temp_object = np.zeros_like(self.object)
        self.rng = np.random.default_rng()
        self.probe_energy = np.max(np.sum(self.data.im_data, axis=(1, 2)))
        self.rows, self.cols = np.indices(self.probe.shape)
        self._algs = {}
        # plt.scatter(self.data.position[:, 1], self.data.position[:, 0])
        # plt.gca().set_aspect('equal')
        # plt.show()

    @abstractmethod
    def run(self, num_iterations, algorithm, animate):
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


class Recon3D(Recon):
    def __init__(self, data: LoadData):
        super().__init__(data)

        self._algs = {}
        pass

    def run(self, num_iterations, algorithm, animate=False):
        pass

    def show_object_and_probe(self):
        pass

    def _apply_update(self, x, y, object_update, probe_update):
        pass

    def _correct_probe(self):
        pass


class Recon2D(Recon):
    def __init__(self, data: LoadData):
        super().__init__(data)
        assert not self.data.is3d, "Cannot reconstruct 3D data with 2D reconstruction algorithms"

        # This is the translation data IN UNITS OF PIXELS
        self.translation = self.data.position / self.pixel_size
        self._correct_probe()
        self.shifted_im_data = np.sqrt(np.fft.ifftshift(self.data.im_data, axes=(1, 2)))

        self._algs = {
            'epie': self._epie,
            'rpie': self._rpie
        }

    def run(self, num_iterations, algorithm, apply_mask_at=(), animate=True):
        try:
            update_function = self._algs[algorithm]
        except KeyError:
            print(f"Warning: '{algorithm}' is not a recognized reconstruction algorithm.")
            return
        fig, ax1, ax2, ax3, ax4 = setup_figure(animate)
        frames = []
        for j in progressbar.progressbar(range(num_iterations)):
            entries = np.arange(self.data.num_entries)
            self.rng.shuffle(entries)
            update_param = j / num_iterations
            for i in entries:
                update_probe = j > 0
                update_function(i, update_param, update_probe)
            if j in apply_mask_at:
                self.object = self.object * self._probe_mask()
            if animate:
                frames.append([ax1.imshow(np.abs(self.object), cmap='bone'),
                               ax2.imshow(np.angle(self.object), cmap='hsv'),
                               ax3.imshow(np.abs(self.probe), cmap='bone'),
                               ax4.imshow(np.angle(self.probe), cmap='hsv')])
        if animate:
            vid = ArtistAnimation(fig, frames, interval=250, repeat_delay=0)

    def _apply_update(self, x, y, object_update, probe_update=None):
        self.object[y + self.rows, x + self.cols] = self.object[y + self.rows, x + self.cols] + object_update
        if probe_update is not None:
            self.probe = self.probe + probe_update
            self._correct_probe()

    def _pre_pie(self, i):
        # Take a slice of the object that's the same size as the probe
        y = int(self.translation[i, 0] + 0.5)
        x = int(self.translation[i, 1] + 0.5)
        region = self.object[y + self.rows, x + self.cols]
        # region = hf.shift(self.object, self.translation[i], crop=self.data.shape[0], subpixel=False)
        psi1 = self.probe * region
        PSI1 = np.fft.fft2(psi1)
        PSI2 = self.shifted_im_data[i] * PSI1 / np.abs(PSI1)
        psi2 = np.fft.ifft2(PSI2)
        d_psi = psi2 - psi1
        return x, y, region, d_psi

    def _epie(self, i: int, param: float, update_probe: bool):
        x, y, region, d_psi = self._pre_pie(i)
        alpha = np.sqrt(1.25-param)
        beta = 1-param
        object_update = alpha * d_psi * np.conj(self.probe) / np.max(np.abs(self.probe)) ** 2
        probe_update = None
        if update_probe:
            probe_update = beta * d_psi * np.conj(region) / np.max(np.abs(region)) ** 2
        self._apply_update(x, y, object_update, probe_update)

    def _rpie(self, i: int, param: float, update_probe: bool):
        x, y, region, d_psi = self._pre_pie(i)
        alpha = 0.05 + 0.45 * param ** 2
        beta = 0.25 + 0.75 * param
        object_update = d_psi * np.conj(self.probe) / \
                        (alpha*np.max(np.abs(self.probe))**2 + (1-alpha)*np.abs(self.probe)**2)
        probe_update = None
        if update_probe:
            probe_update = d_psi * np.conj(region) / \
                           (beta*np.max(np.abs(region))**2 + (1-beta)*np.abs(region)**2)
        self._apply_update(x, y, object_update, probe_update)

    def _correct_probe(self):
        self.probe = self.probe * np.sqrt(self.probe_energy / np.sum(np.abs(self.probe) ** 2)) / self.data.shape[0]
        self.probe = center(self.probe)
        self.probe = np.clip(np.abs(self.probe), 0, np.sort(np.abs(self.probe), axis=None)[-2]) * \
                     np.exp(1j * np.angle(self.probe))

    def _correct_phase(self):
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

        plt.figure()
        obj_length = self.pixel_size * self.object.shape[0]
        pro_length = self.pixel_size * self.probe.shape[0]
        plt.subplot(121, title='Object', xlabel='[microns]', ylabel='[microns]')
        plt.imshow(np.abs(self.object), cmap='bone', extent=(0, obj_length, 0, obj_length))
        plt.subplot(122, title='Probe', xlabel='[microns]', ylabel='[microns]')
        plt.imshow(np.abs(self.probe), cmap='bone', extent=(0, pro_length, 0, pro_length))
        plt.show()


def center(image):
    abs_image = np.abs(image)
    ctr = np.asarray(image.shape) / 2 - 0.5
    c_mass = np.array(ndimage.center_of_mass(abs_image > 0.5*np.max(abs_image)))
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
    probe_array = ndimage.gaussian_filter(np.abs(probe_array),10) * np.exp(1j*np.angle(probe_array))
    # ax = plt.subplot(121, title='Sum of diffraction patterns')
    # plt.imshow(diff_array)
    # plt.subplot(122, sharex=ax, sharey=ax, title='Initial probe guess')
    # plt.imshow(np.abs(probe_array))
    # plt.show()
    return probe_array


def init_probe_flat(shape):
    probe_array = np.zeros(shape)
    probe_center = shape[0] / 2 - 0.5
    probe_radius = shape[0] / 4
    rr, cc = draw.disk((probe_center, probe_center), probe_radius)
    probe_array[rr, cc] = 1
    probe_array = probe_array * np.exp(1j*random(probe_array.shape, 0, 2*np.pi))
    return probe_array


def setup_figure(actually_do_it=True):
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
    recon = Reconstruction('test-2021-07-20.pty', flip_images='y')
    recon.run(10, 'rpie', animate=False)

    # for f_ims in ['', 'x', 'y', 'xy']:
    #     for f_pos in ['', 'x', 'y', 'xy']:
    #         print(f'Diffraction flip: {f_ims}')
    #         print(f'Position flip: {f_pos}')
    #         recon = Reconstruction(LoadData('fake.pty'), flip_images=f_ims, flip_positions=f_pos)
    #         recon.run(15, 'rpie', animate=False)
