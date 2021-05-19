"""
This script is for collecting ptychography data.
"""


import numpy as np
import ptycho_data
import mightex
import micronix
import click


@click.command()
@click.option('-s', '--title', default='test', help='Data will be saved by this name')
@click.option('-d', '--directory', default='C:/Users/jacione/Box/3d-ptycho/data',
              help='Data will be saved in this directory')
@click.option('-2d/-3d', 'is2d', default=False, help='Ignore/include rotation in data collection')
@click.option('-nx', default=10, help='Number of horizontal steps')
@click.option('-ny', default=10, help='Number of vertical steps')
@click.option('-nq', default=2, help='Number of rotational steps')
@click.option('-rx', nargs=2, default=(-4, 4), help='Two numbers: min, max horizontal positions (mm)')
@click.option('-ry', nargs=2, default=(0, 8), help='Two numbers: min, max vertical positions (mm)')
@click.option('-rq', nargs=2, default=(-90, 90), help='Two numbers: min, max rotational positions (deg)')
@click.option('--resolution', nargs=2, default=(2560, 1920), help='Two numbers: horizontal, vertical image size')
@click.option('--distance', default=0.33, help='Sample to detector (m)')
@click.option('--energy', default=1.957346, help='Laser photon energy (eV)')
@click.help_option('-h', '--help')
def collect(title, directory, is2d, nx, ny, nq, rx, ry, rq, resolution, distance, energy):
    """
    CLI for collecting 3D or 2D ptychography data.

    Nick Porter, jacioneportier@gmail.com
    """

    x_range, dx = np.linspace(rx[0], rx[1], nx, retstep=True)
    y_range, dy = np.linspace(ry[0], ry[1], ny, retstep=True)
    q_range, dq = np.linspace(rq[0], rq[1], nq, retstep=True)
    if is2d:
        q_range = np.array([0])
    Y, Q, X = np.meshgrid(y_range, q_range, x_range)
    X = np.ravel(X)
    Y = np.ravel(Y)
    Q = np.ravel(Q)
    num_takes = X.shape[0]

    stages = micronix.MMC200()
    camera = mightex.Camera(resolution=resolution)
    dataset = ptycho_data.DataSet(num_takes=num_takes, title=title, directory=directory, im_shape=resolution,
                                  distance=distance, energy=energy)

    print(f'Taking {num_takes} takes')
    for i in click.progressbar(range(num_takes)):
        stages.set_position((X[i], Y[i], 0, Q[i]))
        diff_pattern = camera.get_frame()
        dataset.record_data(stages.position(), diff_pattern)

    dataset.save_to_cxi()


if __name__ == '__main__':
    collect()
