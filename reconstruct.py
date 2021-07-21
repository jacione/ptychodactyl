"""
Reconstruction code
"""

import click
from experiment.reconstruct_source import Reconstruction


@click.command()
@click.option('-l', '--load', help='Must be a compatible PTY file')
@click.option('-s', '--save', help='Save reconstruction in this location')
@click.option('-a', '--algorithm', type=(str, int), default=[['rpie', 30]], multiple=True,
              help='Algorithm to use in reconstruction, and how many iterations to run (e.g. -a epie 10)')
@click.option('-specs', '--spec_file', default='reconstruction_specs.txt')
def reconstruct(load, save, algorithm, spec_file):
    if load is None:
        load = 'fake.pty'
    elif not load.endswith('.pty'):
        load += '.pty'
    recon = Reconstruction(load, spec_file)

    for alg, n in algorithm:
        recon.run(num_iterations=n, algorithm=alg.lower(), animate=False)

    recon.show_object_and_probe()


if __name__ == '__main__':
    reconstruct()
