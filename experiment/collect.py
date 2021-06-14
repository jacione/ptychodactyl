"""
This script is for collecting ptychography data.
"""


import numpy as np
import ptycho_data
from camera import Mightex, ThorCam
import micronix
import click


@click.command()
@click.help_option('-h', '--help')
@click.option('-v', '--verbose', is_flag=True, default=False, help='Print information at each step')
@click.option('-3d/-2d', 'is3d', default=False, help='Include/ignore rotation in data collection')
@click.option('-s', '--title', default='test', help='Data will be saved by this name')
@click.option('-nx', default=10, help='Number of horizontal steps')
@click.option('-ny', default=10, help='Number of vertical steps')
@click.option('-nq', default=2, help='Number of rotational steps')
@click.option('-rx', nargs=2, default=(-4, 4), help='Two numbers: min, max horizontal positions (mm)')
@click.option('-ry', nargs=2, default=(0, 8), help='Two numbers: min, max vertical positions (mm)')
@click.option('-rq', nargs=2, default=(-90, 90), help='Two numbers: min, max rotational positions (deg)')
@click.option('-fpt', '--frames_per_take', 'frames per take', default=1,
              help='Number of frames to sum for each measurement')
@click.option('--resolution', nargs=2, help='Two numbers: horizontal, vertical image size')
@click.option('-ex', '--exposure', help='Exposure time in milliseconds')
@click.option('-g', '--gain', help='Analog gain')
@click.option('-z', '--distance', default=0.075, help='Sample to detector (m)')
@click.option('--energy', default=1.957346, help='Laser photon energy (eV)')
def collect(title, directory, verbose, is3d, nx, ny, nq, rx, ry, rq, frames_per_take, resolution, exposure, gain,
            distance, energy):
    """
    CLI for collecting 3D or 2D ptychography data.

    Nick Porter, jacioneportier@gmail.com
    """

    print('Beginning ptychography data collection!')

    # Break the horizontal, vertical, and rotational ranges into steps
    X, dx = np.linspace(rx[0], rx[1], nx, retstep=True)
    Y, dy = np.linspace(ry[0], ry[1], ny, retstep=True)
    Q, dq = np.linspace(rq[0], rq[1], nq, retstep=True)
    run_type = '3D ptycho-tomography'
    if not is3d:
        # Ignore default rotation range if -2d flag is passed
        Q = np.array([0])
        nq = 1
        run_type = '2D ptychography'

    # Generate and flatten a meshgrid so that (X[i], Y[i], Q[i]) gives the correct position for the i-th take
    # This is to avoid nested for-loops
    X, Y = np.meshgrid(X, Y)
    X = np.ravel(X)
    Y = np.ravel(Y)
    num_translations = nx * ny
    num_rotations = nq
    num_total = num_rotations * num_translations

    # Initialize the devices and data structures
    stages = micronix.MMC200(verbose=verbose)
    camera = ThorCam(verbose=verbose)
    camera.set_resolution(resolution)
    camera.set_exposure(exposure)
    camera.set_gain(gain)
    if resolution is None:
        resolution = camera.im_shape
    dataset = ptycho_data.CollectData(num_translations=num_translations, num_rotations=num_rotations, title=title,
                                      im_shape=resolution, pixel_size=camera.pixel_size, distance=distance,
                                      energy=energy, verbose=verbose)

    # If it's running verbose, it'll print information on every take. Otherwise, it'll display a simple progress bar.
    print(f'Run type: {run_type}')
    print(f'Camera resolution: {resolution}')
    print(f'Detector distance: {distance} m')
    print(f'Photon energy: {energy} eV')
    print('Positioning (mm):')
    print('     ' + 'MIN'.ljust(6) + 'MAX'.ljust(6) + 'STEPS')
    print('X    ' + f'{rx[0]}'.ljust(6) + f'{rx[1]}'.ljust(6) + f'{nx}'.ljust(6))
    print('Y    ' + f'{ry[0]}'.ljust(6) + f'{ry[1]}'.ljust(6) + f'{ny}'.ljust(6))
    if is3d:
        print('Q    ' + f'{rq[0]}'.ljust(6) + f'{rq[1]}'.ljust(6) + f'{nq}'.ljust(6))
    if verbose:
        counter = lambda N: range(N)
    else:
        counter = lambda N: click.progressbar(range(N))
    print(f'Total: {num_total} takes')

    for i in range(num_rotations):
        print(f'Rotation 1: {Q[i]} deg')
        rotation_complete = False
        with counter(num_translations) as count:
            for j in count:
                stages.set_position((X[j], Y[j], 0, Q[i]))
                diff_pattern = camera.get_frames(frames_per_take)
                rotation_complete = dataset.record_data(stages.get_position(), diff_pattern)
        assert rotation_complete, 'Ran out of translations before dataset was ready to rotate.'

    stages.home_all()
    dataset.save_to_pty()


if __name__ == '__main__':
    collect()
