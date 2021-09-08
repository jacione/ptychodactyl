"""
Main script for collecting ptycho data.

..note::
    Although 3D data collection is possible using this script, the corresponding 3D reconstruction is not yet
    implemented in this library.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from progressbar import progressbar
from ptycho.ptycho_data import CollectData
from ptycho.camera import get_camera
from ptycho.stages import get_stages
from ptycho.scan import xy_scan, r_scan
from ptycho.general import parse_specs


def collect(spec_file='collection_specs.txt', verbose=False):
    """
    CLI for collecting ptycho data. If the whole repository is downloaded, you can just fill out the desired
    parameters in "collection_specs.txt" and run this script from the command line.
    """

    # Load collection parameters from spec file
    specs = parse_specs(f'{spec_file}')
    title = specs['title']
    data_dir = specs['data_dir']
    stage_model = specs['stages']
    pattern = specs['pattern']
    scan_center = specs['scan_center']
    scan_width = specs['scan_width']
    scan_height = specs['scan_height']
    step_size = specs['step_size']
    Z = specs['z_position']
    num_rotations = specs['num_rotations']
    camera_model = specs['camera']
    background_frames = specs['background_frames']
    frames_per_take = specs['frames_per_take']
    resolution = specs['resolution']
    exposure = specs['exposure']
    gain = specs['gain']
    distance = specs['distance']
    energy = specs['energy']

    print('Beginning ptycho data collection!')

    # Generate the scanning positions
    X, Y, num_translations = xy_scan(pattern, scan_center, scan_width, scan_height, step_size)
    Q, num_rotations = r_scan(num_rotations)
    num_total = num_rotations * num_translations
    is3d = num_rotations > 1

    # Initialize the devices and data structures
    stages = get_stages(stage_model, verbose=verbose)
    camera = get_camera(camera_model, verbose=verbose)
    camera.set_frames_per_take(frames_per_take)
    camera.set_resolution(resolution)
    camera.set_exposure(exposure)
    camera.set_gain(gain)
    dataset = CollectData(num_translations=num_translations, num_rotations=num_rotations, title=title,
                          data_dir=data_dir, im_shape=camera.im_shape, pixel_size=camera.pixel_size, distance=distance,
                          energy=energy, verbose=verbose)

    # Print the most important run parameters
    print(f'Run type: {2+is3d}D ptycho')
    print(f'Scan area:')
    print(f'\tPattern:   {pattern.upper()}')
    print(f'\tWidth:     {scan_width:0.4} mm')
    print(f'\tHeight:    {scan_height:0.4} mm')
    print(f'\tStep:      {step_size:0.4} mm')
    if is3d:
        print(f'\tRotations: {num_rotations}')
    print(f'Total: {num_total} takes\n')

    # Record frames for background subtraction
    if background_frames:
        input('Preparing to take background images. Turn laser OFF, then press ENTER to continue...')
        dataset.record_background(camera.get_frames())

    input('Preparing to take ptycho data. Turn laser ON, then press ENTER to continue...')

    for i in range(num_rotations):
        # This is
        print(f'Rotation {i+1}: {Q[i]} deg')
        rotation_complete = False
        for j in progressbar(range(num_translations)):
            if verbose:
                print()
            stages.set_position((X[j], Y[j], Z, Q[i]))
            diff_pattern = camera.get_frames()
            rotation_complete = dataset.record_data(stages.get_position(), diff_pattern)
            # The CollectData.record_data() method should return True only after taking num_translations images.
            # This ensures that the data from each rotation angle stays together.
        assert rotation_complete, 'Ran out of translations before dataset was ready to rotate.'

    # Return the stages to zero
    stages.home_all()

    # Save the data as a .pty (H5) file
    dataset.save_to_pty(cropto=resolution)


if __name__ == '__main__':
    collect()
