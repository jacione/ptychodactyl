"""
Reconstruction code
"""

import click
from experiment.reconstruct_source import Reconstruction
from experiment.ptycho_data import LoadData, GenerateData2D


@click.command()
@click.option('-l', '--load', help='Must be a compatible PTY file')
@click.option('-s', '--save', help='Save reconstruction in this location')
@click.option('-a', '--algorithm', type=(str, int), default=[['rpie', 30]], multiple=True,
              help='Algorithm to use in reconstruction, and how many iterations to run (e.g. -a epie 10')
def reconstruct(load, save, algorithm):
    if load is None:
        load = 'Data/fake.pty'
    recon = Reconstruction(LoadData(load))

    for alg, n in algorithm:
        recon.run(num_iterations=n, algorithm=alg.lower())


if __name__ == '__main__':
    data = GenerateData2D(12, 10000, 'fake', show=False)
    reconstruct()
