import numpy as np
from matplotlib import pyplot as plt
from ptycho.camera import get_camera
from ptycho.stages import get_stages
from ptycho.general import parse_specs
from ptycho.scan import xy_scan
from progressbar import progressbar as pbar
import time


def search(spec_file, obj_size, search_area):
    specs = parse_specs(spec_file)
    title = specs['title']
    camera = get_camera(specs['camera'], temperature=-10)
    stages = get_stages(specs['stages'])

    # Generate a scan for the search area
    # x0 = (0.003264 - stages.zeros['x']) / stages.units
    # y0 = (0.002905 - stages.zeros['y']) / stages.units
    x0 = 0.300  # Center in mm
    y0 = 0.800  # center in mm
    X, Y, N = xy_scan('hex', (x0, y0), search_area[0], search_area[1], obj_size/2, False)

    # Scan over the full range. At each position, take a quick image and record only the sum
    search_results = np.zeros((N, 3))
    print('Scanning object space for features...')
    time.sleep(0.1)
    for i in pbar(range(N)):
        try:
            stages.set_position((X[i], Y[i], 0, 0))
            res = np.mean(camera.get_frames())
            # res = np.exp(-(X[i]**2 + Y[i]**2))
            search_results[i] = [X[i], Y[i], res]
        except IOError:
            search_results = search_results[:i]
            break

    # Save the positions and sums as a npy file
    np.save(f'../data/{title}_search.npy', search_results)

    # Surface plot the positions and sums to see outliers
    plt.tripcolor(X, Y, search_results[:, -1], cmap='inferno')
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.show()
    return


def analyze_search(search_file):
    data = np.load(f'../data/{search_file}').T
    # Surface plot the positions and sums to see outliers
    X = data[0]
    Y = data[1]
    Z = data[2]
    plt.tripcolor(X, Y, Z, cmap='inferno')
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    search('../ptycho/collection_specs.txt', 0.06, (1.2, 1.2))
    # analyze_search('test_search.npy')
