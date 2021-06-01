import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.misc import ascent, face
from skimage.restoration import unwrap_phase
from time import perf_counter


def fft(image):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))


def ifft(image):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(image)))


def show(image):
    plt.figure(tight_layout=True)
    plt.subplot(111, xticks=[], yticks=[])
    plt.imshow(image)
    plt.show()


def random(size, low=0.0, high=1.0):
    rng = np.random.default_rng()
    return low + (high-low)*rng.random(size)


def alt_shift(arr, shift_amt, crop=None):
    amp = ndimage.shift(np.abs(arr), shift_amt)
    phi = ndimage.shift(unwrap_phase(np.angle(arr)), shift_amt)
    new_arr = amp * np.exp(1j * phi)
    if crop is not None:
        new_arr = new_arr[:crop, :crop]
    return new_arr


def shift(arr, shift_amt, crop=None):
    amp = ndimage.shift(np.abs(arr), shift_amt)
    phi = np.angle(ndimage.shift(arr, shift_amt))
    new_arr = amp * np.exp(1j * phi)
    if crop is not None:
        new_arr = new_arr[:crop, :crop]
    return new_arr


def demo_image():
    x = 50
    y = 400
    f = face(gray=True)[x:x+512, y:y+512]
    f = f / np.max(f)
    a = ascent() / np.max(ascent())
    img = f * np.exp(10j*np.pi*a)
    return img


def demo_shift():
    img = demo_image()
    sft1, err1, dt1 = test_shift(ndimage.shift)
    sft2, err2, dt2 = test_shift(shift)
    sft3, err3, dt3 = test_shift(alt_shift)
    print(''.ljust(12) + 'Scipy'.ljust(12) + 'Wrapped'.ljust(12) + 'Unwrapped'.ljust(12))
    print('Error'.ljust(12) + f'{err1}'.ljust(12) + f'{err2}'.ljust(12) + f'{err3}'.ljust(12))
    print('Time (s)'.ljust(12) + f'{dt1}'.ljust(12) + f'{dt2}'.ljust(12) + f'{dt3}'.ljust(12))
    sbpt = {'xticks': [], 'yticks': []}
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='all', sharey='all', subplot_kw=sbpt, tight_layout=True)
    ax1.set_title('Original')
    ax2.set_title('Scipy')
    ax3.set_title('Wrapped')
    ax4.set_title('Unwrapped')
    ax1.imshow(np.abs(img), clim=[0, 1])
    ax2.imshow(np.abs(sft1), clim=[0, 1])
    ax3.imshow(np.abs(sft2), clim=[0, 1])
    ax4.imshow(np.abs(sft3), clim=[0, 1])
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='all', sharey='all', subplot_kw=sbpt, tight_layout=True)
    ax1.set_title('Original')
    ax2.set_title('Scipy')
    ax3.set_title('Wrapped')
    ax4.set_title('Unwrapped')
    ax1.imshow(np.angle(img))
    ax2.imshow(np.angle(sft1))
    ax3.imshow(np.angle(sft2))
    ax4.imshow(np.angle(sft3))
    plt.show()


def detect(arr, saturation=1.0, bitdepth=0, seed=None):
    arr = np.array(arr)
    signal_max = np.max(arr)
    pixel_max = 2 ** bitdepth - 1
    if seed is None:
        seed = np.random.randint(2 ** 31)
    rng = np.random.default_rng(seed)

    # Apply saturation
    print(f'\tSaturation: {int(100 * saturation)}%')
    arr = np.clip(arr * saturation, None, signal_max)

    if bitdepth != 0:
        # Scale data to bit depth:
        print(f'\tBit depth: {bitdepth}-bit')
        arr = np.around(arr * pixel_max / signal_max)

        # Apply bit reduction
        arr = arr.astype('u2')

        # Apply noise
        print(f'\tNoise model: Poisson')
        print(f'\tNoise seed: {seed}')
        # lam = len(np.unique(arr))
        # lam = 2**np.ceil(np.log2(lam))
        # arr = rng.poisson(arr*lam) / float(lam)
        arr = rng.poisson(arr+10)

        arr = np.clip(arr, None, pixel_max)

    return arr


def calc_error(actual, altered, border=5):
    b = border
    gamma = (np.sum(actual[b:-b, b:-b] * np.conj(altered)[b:-b, b:-b]) /
             np.sum(np.abs(altered)[b:-b, b:-b] ** 2))
    error = (np.sum(np.abs(actual[b:-b, b:-b] - gamma * altered[b:-b, b:-b]) ** 2) /
             np.sum(np.abs(actual[b:-b, b:-b]) ** 2))
    return error


def test_shift(func):
    img = demo_image()
    t0 = perf_counter()
    sft = func(img, 2.5)
    sft = func(sft, -4.2)
    sft = func(sft, 1.7)
    t1 = perf_counter()
    err = np.around(calc_error(img, sft, 15), 4)
    dt = np.around(t1 - t0, 4)
    return sft, err, dt


if __name__ == '__main__':
    demo_shift()
