"""
This script is for collecting ptychography data.
"""


import click
from experiment.ptycho_data import CollectData
from experiment.camera import Mightex, ThorCam
from experiment.micronix import MMC200
from experiment.scan import xy_scan, r_scan


@click.command()
@click.help_option('-h', '--help')
@click.option('-v', '--verbose', is_flag=True, default=False, help='Print information at each step')
@click.option('-s', '--title', default='test', help='Data will be saved by this name')
@click.option('-w', '--width', default=10, help='Horizontal scanning range (mm)')
@click.option('-h', '--height', default=10, help='Vertical scanning range (mm)')
@click.option('-d', '--step_size', default=0.5, help='Step size (mm)')
@click.option('-p', '--pattern', default='rect', help='Geometric pattern for ptychography scan')
@click.option('-r', '--num_rotations', default=0, help='Number of rotational steps (zero for no rotation)')
@click.option('-fpt', '--frames_per_take', 'frames per take', default=1,
              help='Number of frames to sum for each measurement')
@click.option('-res', '--resolution', nargs=2, help='Desired image side length (in pixels)')
@click.option('-exp', '--exposure', help='Exposure time in milliseconds')
@click.option('-gain', '--gain', help='Analog gain')
@click.option('-dist', '--distance', default=0.075, help='Sample to detector (m)')
@click.option('--energy', default=1.957346, help='Laser photon energy (eV)')
def collect(title, verbose, width, height, step_size, pattern, num_rotations, frames_per_take, resolution, exposure,
            gain, distance, energy):
    """
    CLI for collecting 3D or 2D ptychography data.

    Nick Porter, jacioneportier@gmail.com
    """

    print('Beginning ptychography data collection!')

    # Generate the scanning positions
    X, Y, num_translations = xy_scan(pattern, width, height, step_size)
    Q, num_rotations = r_scan(num_rotations)
    num_total = num_rotations * num_translations
    is3d = num_rotations > 1

    # Initialize the devices and data structures
    stages = MMC200(verbose=verbose)
    camera = ThorCam(verbose=verbose)
    camera.set_resolution(resolution)
    camera.set_exposure(exposure)
    camera.set_gain(gain)
    if resolution is None:
        resolution = camera.im_shape
    dataset = CollectData(num_translations=num_translations, num_rotations=num_rotations, title=title,
                          im_shape=resolution, pixel_size=camera.pixel_size, distance=distance, energy=energy,
                          verbose=verbose)

    if is3d:
        print(f'Run type: 3D ptychography')
    else:
        print('Run type: 2D ptychography')
    print(f'Scan area:')
    print(f'\tScan pattern: {pattern.upper()}')
    print(f'\tWidth:     {width:0.4} mm')
    print(f'\tHeight:    {height:0.4} mm')
    print(f'\tStep size: {step_size:0.4} mm')
    if is3d:
        print(f'\tRotations: {num_rotations}')
    print(f'Total: {num_total} takes')

    # If it's running verbose, it'll print information on every take. Otherwise, it'll display a simple progress bar.
    if verbose:
        def counter(N): range(N)
    else:
        def counter(N): click.progressbar(range(N))

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
