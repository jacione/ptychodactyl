import numpy as np
from matplotlib import pyplot as plt
from camera import get_camera
from stages import get_stages
from utils.general import parse_specs
from scan import xy_scan
from progressbar import progressbar as pbar
import time


def search(spec_file, obj_size, search_area):
    specs = parse_specs(spec_file)
    title = specs['title']
    camera = get_camera(specs['camera'])
    stages = get_stages(specs['stages'])

    # Generate a scan for the search area
    # x0 = (0.003264 - stages.zeros['x']) / stages.units
    # y0 = (0.002905 - stages.zeros['y']) / stages.units
    x0 = 0.581
    y0 = 0.373
    X, Y, N = xy_scan('hex', (x0, y0), search_area[0], search_area[1], obj_size/2, False)

    # Scan over the full range. At each position, take a quick image and record only the sum
    search_results = np.zeros((N, 3))
    print('Scanning object space for features...')
    time.sleep(0.1)
    for i in pbar(range(N)):
        stages.set_position((X[i], Y[i], 0, 0))
        res = np.mean(camera.get_frames())
        # res = np.exp(-(X[i]**2 + Y[i]**2))
        search_results[i] = [X[i], Y[i], res]

    # Save the positions and sums as a npy file
    np.save(f'data/{title}_search.npy', search_results)

    # Surface plot the positions and sums to see outliers
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(X, Y, search_results[:, -1], cmap='inferno')
    plt.show()
    return


def analyze_search(search_file):
    data = np.load(f'data/{search_file}').T
    # Surface plot the positions and sums to see outliers
    X = data[0][data[1] > 0.2]
    Y = data[1][data[1] > 0.2]
    Z = data[2][data[1] > 0.2]
    # X = data[0]
    # Y = data[1]
    # Z = data[2]
    plt.tripcolor(X, Y, Z, cmap='inferno')
    plt.gca().set_aspect('equal')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap='inferno')
    plt.show()


if __name__ == '__main__':
    search('collection_specs.txt', 0.02, (0.1, 0.1))
    # analyze_search('test_search.npy')
