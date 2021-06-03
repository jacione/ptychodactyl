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
@click.option('-d', '--directory', default='C:/Users/jacione/Box/3d-ptycho/data',
              help='Data will be saved in this directory')
@click.option('-nx', default=10, help='Number of horizontal steps')
@click.option('-ny', default=10, help='Number of vertical steps')
@click.option('-nq', default=2, help='Number of rotational steps')
@click.option('-rx', nargs=2, default=(-4, 4), help='Two numbers: min, max horizontal positions (mm)')
@click.option('-ry', nargs=2, default=(0, 8), help='Two numbers: min, max vertical positions (mm)')
@click.option('-rq', nargs=2, default=(-90, 90), help='Two numbers: min, max rotational positions (deg)')
@click.option('--resolution', nargs=2, help='Two numbers: horizontal, vertical image size')
@click.option('--exposure', help='Exposure time in milliseconds')
@click.option('--gain', help='Analog gain')
@click.option('--distance', default=0.33, help='Sample to detector (m)')
@click.option('--energy', default=1.957346, help='Laser photon energy (eV)')
def collect(title, directory, verbose, is3d, nx, ny, nq, rx, ry, rq, resolution, exposure, gain, distance, energy):
    """
    CLI for collecting 3D or 2D ptychography data.

    Nick Porter, jacioneportier@gmail.com
    """

    print('Beginning ptychography data collection!')

    # Break the horizontal, vertical, and rotational ranges into steps
    x_range, dx = np.linspace(rx[0], rx[1], nx, retstep=True)
    y_range, dy = np.linspace(ry[0], ry[1], ny, retstep=True)
    q_range, dq = np.linspace(rq[0], rq[1], nq, retstep=True)
    run_type = '3D ptycho-tomography'
    if is3d:
        # Ignore default rotation range if -2d flag is passed
        q_range = np.array([0])
        run_type = '2D ptychography'

    # Generate and flatten a meshgrid so that (X[i], Y[i], Q[i]) gives the correct position for the i-th take
    # This is to avoid nested for-loops
    Y, Q, X = np.meshgrid(y_range, q_range, x_range)
    X = np.ravel(X)
    Y = np.ravel(Y)
    Q = np.ravel(Q)
    num_takes = X.shape[0]

    # Initialize the devices and data structures
    stages = micronix.MMC200(verbose=verbose)
    camera = Mightex(verbose=verbose)
    camera.set_resolution(resolution)
    camera.set_exposure(exposure)
    camera.set_gain(gain)
    dataset = ptycho_data.DataSet(num_takes=num_takes, title=title, directory=directory, im_shape=resolution,
                                  pixel_size=camera.pixel_size, distance=distance, energy=energy, is3d=is3d,
                                  verbose=verbose)

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
    print(f'Total: {num_takes} takes')

    with counter(num_takes) as count:
        for i in count:
            stages.set_position((X[i], Y[i], 0, Q[i]))
            diff_pattern = camera.get_frames()
            dataset.record_data(stages.get_position(), diff_pattern)

    stages.home_all()
    dataset.save_to_cxi()


if __name__ == '__main__':
    collect()
