"""
Reconstruction code
"""

import numpy as np
from matplotlib import pyplot as plt
import h5py
import click


@click.command()
@click.option('-l', '--load', nargs=2, help='Must be a compatible CXI file')
@click.option('-s', '--save', help='Save reconstruction in this location')
def reconstruct(load, save):
    print(load)
    print(save)
    pass


if __name__ == '__main__':
    reconstruct()
