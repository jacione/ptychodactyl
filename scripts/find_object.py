import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from matplotlib import pyplot as plt
from ptycho.camera import get_camera
from ptycho.stages import get_stages
from ptycho.scan import xy_scan
from ptycho.specs import CollectionSpecs
from tqdm import tqdm


SPECS = CollectionSpecs(
    # General parameters - these MUST be given values.
    title='test',
    stages='attocube',
    camera='andor',

    # Scan parameters
    scan_center=(0.300, 0.600),
    scan_width=1.2,
    scan_height=1.2,
    scan_step=0.03,
    z_position=-2.0
)


def search(specs, obj_size, search_area):
    title = specs.title
    camera = get_camera(specs.camera)
    stages = get_stages(specs.stages)

    # Generate a scan for the search area
    # x0 = (0.003264 - stages.zeros['x']) / stages.units
    # y0 = (0.002905 - stages.zeros['y']) / stages.units
    X, Y, N = xy_scan('hex', specs.scan_center, specs.scan_width, specs.scan_height, specs.scan_step, False)

    # Scan over the full range. At each position, take a quick image and record only the sum
    search_results = np.zeros((N, 3))
    print('Scanning object space for features...')
    time.sleep(0.1)
    for i in tqdm(range(N)):
        try:
            stages.set_position((X[i], Y[i], specs.z_position, 0))
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
    search(SPECS, 0.06, (1.2, 1.2))
    # analyze_search('test_search.npy')
