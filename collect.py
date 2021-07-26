"""
This script is for collecting ptychography data.
"""


import click
from experiment.ptycho_data import CollectData
from experiment.camera import ThorCam
from experiment.micronix import MMC200
from experiment.scan import xy_scan, r_scan
from experiment.utils.helper_funcs import parse_specs


@click.command()
@click.help_option('-h', '--help')
@click.option('-v', '--verbose', is_flag=True, default=False, help='Print information at each step')
@click.option('--spec_file', default='collection_specs.txt')
def collect(verbose, spec_file):
    """
    CLI for collecting 3D or 2D ptychography data.

    Nick Porter, jacioneportier@gmail.com
    """

    specs = parse_specs(f'{spec_file}')
    title = specs['title']
    pattern = specs['pattern']
    width = specs['width']
    height = specs['height']
    step_size = specs['step_size']
    num_rotations = specs['num_rotations']
    background_frames = specs['background_frames']
    frames_per_take = specs['frames_per_take']
    resolution = specs['resolution']
    exposure = specs['exposure']
    gain = specs['gain']
    distance = specs['distance']
    energy = specs['energy']

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
    dataset = CollectData(num_translations=num_translations, num_rotations=num_rotations, title=title,
                          im_shape=camera.im_shape, pixel_size=camera.pixel_size, distance=distance, energy=energy,
                          verbose=verbose)

    if is3d:
        print(f'Run type: 3D ptychography')
    else:
        print('Run type: 2D ptychography')
    print(f'Scan area:')
    print(f'\tPattern:   {pattern.upper()}')
    print(f'\tWidth:     {width:0.4} mm')
    print(f'\tHeight:    {height:0.4} mm')
    print(f'\tStep:      {step_size:0.4} mm')
    if is3d:
        print(f'\tRotations: {num_rotations}')
    print(f'Total: {num_total} takes\n')

    if background_frames:
        input('Preparing to take background images. Turn laser OFF, then press ENTER to continue...')
        dataset.record_background(camera.get_frames(frames_per_take))

    input('Preparing to take ptychography data. Turn laser ON, then press ENTER to continue...')
    for i in range(num_rotations):
        print(f'Rotation 1: {Q[i]} deg')
        rotation_complete = False
        with click.progressbar(range(num_translations)) as count:
            for j in count:
                if verbose:
                    print()
                stages.set_position((X[j], Y[j], 0, Q[i]))
                diff_pattern = camera.get_frames(frames_per_take)
                rotation_complete = dataset.record_data(stages.get_position(), diff_pattern)
        assert rotation_complete, 'Ran out of translations before dataset was ready to rotate.'

    stages.home_all()
    dataset.save_to_pty(cropto=resolution)


if __name__ == '__main__':
    collect()
