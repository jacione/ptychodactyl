"""
Reconstruction code
"""

import click
from experiment.reconstruct_source import Reconstruction
from experiment.utils.helper_funcs import parse_specs


LOADATA_KW = ['flip_images', 'flip_positions', 'background_subtract', 'vbleed_correct', 'threshold']
RUN_KW = ['algorithm', 'num_iterations', 'obj_up_initial', 'obj_up_final', 'pro_up_initial', 'pro_up_final']


@click.command()
@click.option('--spec_file', default='reconstruction_specs.txt')
def reconstruct(spec_file):

    specs = parse_specs(spec_file)

    load = specs['load']
    if not load.endswith('.pty'):
        load = load + '.pty'

    recon = Reconstruction(load, **{kw: specs[kw] for kw in LOADATA_KW})

    for n in range(specs['num_stages']):
        recon.run({kw: specs[kw][n] for kw in RUN_KW}, animate=specs['animate'])

    recon.save()

    recon.show_object_and_probe()


if __name__ == '__main__':
    reconstruct()
