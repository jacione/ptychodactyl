"""
Some utilitarian functions for plotting. Also, a few demos.
"""
from matplotlib import colors
from matplotlib.animation import ArtistAnimation
from scipy import misc
from ptycho.general import *


here = os.path.dirname(os.path.realpath(__file__))


def show(image):
    """Quick display of an image with some common options preloaded"""
    plt.figure(tight_layout=True)
    plt.subplot(111, xticks=[], yticks=[])
    plt.imshow(image, cmap='plasma')
    plt.show()


def comp_to_rgb(comp_img):
    amp = normalize(np.abs(comp_img))
    phi = normalize(np.angle(comp_img))
    one = np.ones_like(amp)
    hsv = np.dstack((phi, one, amp))
    return colors.hsv_to_rgb(hsv)


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


def demo_ft():
    pic = misc.ascent()
    ft_pic = fft(pic)
    amplitude_squared = np.log(np.abs(ft_pic))
    phase = np.angle(ft_pic)
    amplitude_squared = normalize(amplitude_squared)
    phase = normalize(phase)
    ones = np.ones_like(pic)
    comp_image = np.dstack((phase, ones, amplitude_squared))
    comp_image = colors.hsv_to_rgb(comp_image)

    bad_pic = np.fft.ifftn(np.abs(np.fft.fftn(pic)))
    bad_amp = normalize(np.log(np.abs(bad_pic)))
    bad_phi = normalize(np.angle(bad_pic))
    # bad_comp = np.dstack((bad_phi, ones, bad_amp))
    # bad_comp = colors.hsv_to_rgb(bad_comp)

    plt.subplot(131, xticks=[], yticks=[])
    plt.imshow(pic, cmap='gray')
    plt.subplot(132, xticks=[], yticks=[])
    plt.imshow(amplitude_squared, cmap='gray')
    plt.subplot(133, xticks=[], yticks=[])
    plt.imshow(bad_amp, cmap='gray')
    plt.tight_layout()
    plt.show()
    plt.imsave(f'{here}/pic.png', pic, cmap='gray')
    plt.imsave(f'{here}/ft_pic.png', comp_image)
    plt.imsave(f'{here}/intensity.png', amplitude_squared, cmap='gray')
    plt.imsave(f'{here}/bad_pic.png', bad_amp, cmap='gray')


def demo_phase():
    x = np.linspace(-2*np.pi, 2*np.pi, 500)

    n_frames = 50
    adjust = np.sin(np.linspace(0, 2*np.pi, n_frames))
    amp = 1.25 + adjust
    phi = 2*np.pi*adjust

    fig = plt.figure(figsize=(6, 7), tight_layout=True)
    frames = []
    ax1 = plt.subplot(211, xticks=[], yticks=[], title='Amplitude')
    ax2 = plt.subplot(212, xticks=[], yticks=[], title='Phase')
    for i in range(n_frames):
        f1, = ax1.plot(amp[i]*np.exp(-x**2/4)*np.sin(10*x), c='b', animated=True)
        f2, = ax2.plot(np.exp(-x**2/4)*np.sin(10*x - phi[i]), c='b', animated=True)
        frames.append([f1, f2])
    vid = ArtistAnimation(fig, frames, interval=100, repeat_delay=0)
    vid.save(f'{here}/amp_phase.gif', fps=10)
    plt.show()


def demo_double_pinhole():
    N = 1024
    ctr = N // 2
    d = N // 16
    r = 16
    obj = np.zeros((N, N))
    obj[draw.disk((ctr, ctr-d), r)] = 1
    obj[draw.disk((ctr, ctr+d), r)] = 1
    # obj[draw.disk((ctr+d, ctr), r)] = 1
    # obj[draw.disk((ctr-d, ctr), r)] = 1

    img = fft(obj)
    img = np.abs(img)

    plt.subplot(121, xticks=[], yticks=[])
    plt.imshow(obj[1*N//3:2*N//3, 1*N//3:2*N//3], cmap='plasma')
    plt.subplot(122, xticks=[], yticks=[])
    plt.imshow(img[1*N//3:2*N//3, 1*N//3:2*N//3], cmap='plasma')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass
