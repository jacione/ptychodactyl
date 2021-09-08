"""
Main script for reconstructing ptycho data.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ptycho.recon import Reconstruction
from ptycho.general import parse_specs


LOADATA_KW = ['flip_images', 'flip_positions', 'rotate_positions', 'background_subtract', 'vbleed_correct', 'threshold']
RUN_KW = ['algorithm', 'num_iterations', 'obj_up_initial', 'obj_up_final', 'pro_up_initial', 'pro_up_final']


def reconstruct(spec_file='reconstruction_specs.txt'):
    """
    CLI for reconstructing ptycho data. If the whole repository is downloaded, you can just fill out the desired
    parameters in "reconstruction_specs.txt" and run this script from the command line.
    """

    specs = parse_specs(spec_file)

    title = specs['title']
    data_dir = specs['data_dir']
    if not title.endswith('.pty'):
        title = title + '.pty'
    if data_dir == '':
        title = Path(__file__).parents[1] / 'data' / title
    else:
        title = Path(data_dir) / title

    recon = Reconstruction(title, **{kw: specs[kw] for kw in LOADATA_KW})

    for n in range(specs['num_stages']):
        recon.run({kw: specs[kw][n] for kw in RUN_KW}, animate=specs['animate'])

    recon.save()

    recon.show_object_and_probe()


if __name__ == '__main__':
    reconstruct()
