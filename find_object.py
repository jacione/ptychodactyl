import numpy as np
from matplotlib import pyplot as plt
from camera import get_camera
from stages import get_stages
from utils.general import parse_specs
from scan import hex_scan
from progressbar import progressbar as pbar


def search(spec_file, obj_size, search_area):
    specs = parse_specs(spec_file)
    title = specs['title']
    camera = get_camera(specs['camera'])
    stages = get_stages(specs['stages'])

    # Generate a scan for the search area
    X, Y, N = hex_scan((0, 0), search_area[0], search_area[1], obj_size/2, False)

    # Scan over the full range. At each position, take a quick image and record only the sum
    search_results = np.zeros((N, 3))
    print('Scanning object space for features...')
    for i in pbar(range(N)):
        stages.set_position((X[i], Y[i], 0, 0))
        res = np.sum(camera.get_frames())
        # res = np.exp(-(X[i]**2 + Y[i]**2))
        search_results[i] = [X[i], Y[i], res]

    # Save the positions and sums as a npy file
    np.save(f'data/{title}_search.npy', search_results)

    # Surface plot the positions and sums to see outliers
    plt.tripcolor(X, Y, search_results[:, -1])
    plt.gca().set_aspect('equal')
    plt.show()
    return


if __name__ == '__main__':
    search('collection_specs.txt', 0.05, (1.0, 1.0))
