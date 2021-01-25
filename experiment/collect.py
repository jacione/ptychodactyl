"""
This script is for collecting ptychography data.
"""


import numpy as np
import h5py
import os


def collect_manual():
    print('Beginning manual ptychography...')

    savedir = 'C:/Users/jacio/OneDrive/Escritorio/test_data'
    npydir = savedir + '/temp'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    if not os.path.exists(npydir):
        os.mkdir(npydir)
    filename = input('Save data as: ')
    if filename.endswith('.cxi'):
        filename = filename[:-4]
    print(filename)

    def help():
        print()
        print('When asked for input, please adjust the sample position along ONE axis, then input the adjustment in '
              'the form "(axis)(position in mm)", for example:\n'
              '\tx1.5\t\tset the x (horizontal) position to 1.5 mm\n'
              '\ty2.1\t\tset the y (vertical) position to 2.1 mm')
        print('You may also input any of the following commands:')
        print('\th\t\t\tprint this help menu\n'
              '\tq\t\t\tsave the dataset and end the program\n')
        print()

    help()
    x = 0
    y = 0
    n = 0
    input('Set sample to initial (zero) position, then press ENTER...')
    n = take(x, y, n, savedir)

    while True:
        cmd = input('Input: ')
        if cmd == '':
            continue
        elif cmd[0] == 'x':
            try:
                x = float(cmd[1:])
                n = take(x, y, n, npydir)
            except ValueError:
                print('\tDATA NOT TAKEN! (could not parse input)')
        elif cmd[0] == 'y':
            try:
                y = float(cmd[1:])
                n = take(x, y, n, npydir)
            except ValueError:
                print('\tDATA NOT TAKEN! (could not parse input)')
        elif cmd == 'h':
            help()
        elif cmd == 'q':
            npy_to_cxi(savedir, filename)
            break

    quit()
    pass


def collect_auto():
    pass


def take(x, y, n, savedir):
    print(f'\tData taken: \t\tx={x}\ty={y}\tn={n}')
    return n+1


def npy_to_cxi(savedir, filename):
    pass
    # Save the data to a CXI file
    f = h5py.File(f'{savedir}/{filename}.cxi', 'w')
    entry = f.create_group('entry_1')

    data = entry.create_group('data_1')
    data.create_dataset('data', data=diff_patterns, dtype='int')
    data.create_dataset('translation', data=translation)

    inst = entry.create_group('instrument_1')
    source = inst.create_group('source_1')
    source.create_dataset('energy', data=energy * 1.602176634e-19)
    detector = inst.create_group('detector_1')
    detector.create_dataset('distance', data=z_dist)
    detector.create_dataset('x_pixel_size', data=pixel_size * 10 ** -6)
    detector.create_dataset('y_pixel_size', data=pixel_size * 10 ** -6)


if __name__ == '__main__':
    collect_manual()