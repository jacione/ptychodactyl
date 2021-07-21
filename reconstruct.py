"""
Reconstruction code
"""

import click
from experiment.reconstruct_source import Reconstruction
from experiment.utils.helper_funcs import parse_specs
from experiment.ptycho_data import LoadData


LOADATA_KW = ['flip_images', 'flip_positions', 'background_subtract', 'vbleed_correct', 'threshold']


@click.command()
@click.option('-a', '--algorithm', type=(str, int), default=[['rpie', 30]], multiple=True,
              help='Algorithm to use in reconstruction, and how many iterations to run (e.g. -a epie 10)')
@click.option('-specs', '--spec_file', default='reconstruction_specs.txt')
def reconstruct(algorithm, spec_file):

    specs = parse_specs(spec_file)

    load = specs['load']
    if not load.endswith('.pty'):
        load = load + '.pty'
    saveas = specs['saveas']

    recon = Reconstruction(load, **{kw: specs[kw] for kw in LOADATA_KW})

    for alg, n in algorithm:
        recon.run(num_iterations=n, algorithm=alg.lower(), animate=False)

    recon.save()

    recon.show_object_and_probe()


if __name__ == '__main__':
    reconstruct()
