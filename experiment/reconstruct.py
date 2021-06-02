"""
Reconstruction code
"""

import click
from reconstruct_source import CXIData, Reconstruction


@click.command()
@click.option('-l', '--load', help='Must be a compatible CXI file')
@click.option('-s', '--save', help='Save reconstruction in this location')
@click.option('-alg', '--algorithm', default='epie', help='Algorithm to use in reconstruction')
def reconstruct(load, save, algorithm):
    if load is None:
        load = 'libs/dummy_data/fake.cxi'
    recon = Reconstruction(CXIData(load))
    recon.run(5, algorithm, False)
    recon.show_object_and_probe()


if __name__ == '__main__':
    reconstruct()
