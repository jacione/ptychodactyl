import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy import misc
import os


here = os.path.dirname(os.path.realpath(__file__))


def normalize(x):
    return (x-np.min(x)) / (np.max(x) - np.min(x))


def demo_ft():
    pic = misc.ascent()
    ft_pic = np.fft.fftshift(np.fft.fftn(pic))
    intensity = np.log(np.abs(ft_pic))
    phase = np.angle(ft_pic)
    intensity = normalize(intensity)
    phase = normalize(phase)
    ones = np.ones_like(pic)
    comp_image = np.dstack((phase, ones, intensity))
    comp_image = colors.hsv_to_rgb(comp_image)

    bad_pic = np.fft.ifftn(np.abs(np.fft.fftn(pic)))
    bad_amp = normalize(np.log(np.abs(bad_pic)))
    bad_phi = normalize(np.angle(bad_pic))
    bad_comp = np.dstack((bad_phi, ones, bad_amp))
    bad_comp = colors.hsv_to_rgb(bad_comp)

    plt.subplot(131, xticks=[], yticks=[])
    plt.imshow(pic, cmap='gray')
    plt.subplot(132, xticks=[], yticks=[])
    plt.imshow(intensity, cmap='gray')
    plt.subplot(133, xticks=[], yticks=[])
    plt.imshow(bad_amp, cmap='gray')
    plt.tight_layout()
    plt.show()
    plt.imsave(f'{here}/pic.png', pic, cmap='gray')
    plt.imsave(f'{here}/ft_pic.png', comp_image)
    plt.imsave(f'{here}/intensity.png', intensity, cmap='gray')
    plt.imsave(f'{here}/bad_pic.png', bad_amp, cmap='gray')


def phase_wheel():
    lin = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(lin, lin)
    amp = np.abs(X+Y*1j)
    phi = normalize(np.angle(X+Y*1j))
    ones = np.ones_like(amp)

    wheel = np.dstack((phi, ones, amp))
    wheel[amp > 1] = [0, 0, 1]
    wheel = colors.hsv_to_rgb(wheel)

    plt.subplot(111, xticks=[], yticks=[])
    plt.imshow(wheel)
    plt.text(205, 100, r'$0$', va='center', ha='left', size=24)
    plt.text(-5, 100, r'$\pi$', va='center', ha='right', size=24)
    plt.text(100, 0, r'$\pi/2$', va='bottom', ha='center', size=24)
    plt.text(100, 200, r'$3\pi/2$', va='top', ha='center', size=24)
    plt.tight_layout()
    plt.box(False)
    plt.show()


def demo_phase():
    x = np.linspace(-np.pi, np.pi, 500)
    phi = np.pi / 4
    y1 = lambda n: 0.25 * (n+1) * np.exp(-0.75*x**2) * np.sin(6 * x)
    g1 = lambda n: 0.25 * (n+1) * np.exp(-0.75*x**2)
    y2 = lambda n: np.exp(-0.75*x**2) * np.sin(10 * (x - n*phi))

    x0 = np.array([0, 1, 1, 0]) * 2*np.pi
    y0 = np.array([0, 0, 1, 1]) * 1.3

    ax1 = plt.subplot(121, xticks=[], yticks=[], title='Different Amplitudes')
    ax2 = plt.subplot(122, xticks=[], yticks=[], title='Different Phases')
    for i in range(4):
        ax1.plot(x0[i]+x, y0[i]+y1(i))
        ax1.plot(x0[i]+x, y0[i]+g1(i), lw=1, ls='--', c='k')
        ax1.plot(x0[i]+x, y0[i]-g1(i), lw=1, ls='--', c='k')
        ax2.plot(x, 2*i+y2(i))

    ax2.axvline(0, c='k', lw=1, ls='--')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    demo_phase()
