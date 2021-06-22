import numpy as np
import matplotlib.pyplot as plt
import experiment.helper_funcs as hf


def xy_scan(style, width, height, step):
    style_dict = {'rect': rect_scan, 'hex': hex_scan, 'spiral': spiral_scan}
    scan = style_dict[style]
    return scan(width, height, step)
    

def rect_scan(width, height, step):
    x = np.arange(-width/2, width/2, step)
    y = np.arange(-height/2, height/2, step)
    X, Y = np.meshgrid(x, y)
    dxy = 0.05*step
    X = np.ravel(X + hf.random(X.shape, -dxy, dxy))
    Y = np.ravel(Y + hf.random(Y.shape, -dxy, dxy))
    N = len(X)
    return X, Y, N


def hex_scan(width, height, step):
    x = np.arange(-width/2, width/2, step)
    y = np.arange(-height/2, height/2, step*np.sqrt(3)/2)
    X, Y = np.meshgrid(x, y)
    for i in range(X.shape[0]):
        if i % 2:
            X[i] = X[i] + step / 2
    dxy = 0.05*step
    X = np.ravel(X + hf.random(X.shape, -dxy, dxy))
    Y = np.ravel(Y + hf.random(Y.shape, -dxy, dxy))
    N = len(X)
    return X, Y, N


def spiral_scan(width, height, step):
    r_max = np.sqrt(width**2 + height**2) / 2
    coords = np.array([-step/5, step/3.3])
    r = step/4
    phi = 0
    while r < r_max:
        phi = phi + (step / r)
        r = phi * step / (2*np.pi)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        coords = np.vstack((coords, [x, y]))
    X = coords[:, 0]
    Y = coords[:, 1]
    bounds = (X < width/2) * (X > -width/2) * (Y < height/2) * (Y > -height/2)
    X = X[bounds]
    Y = Y[bounds]
    N = len(X)
    return X, Y, N


def r_scan(num_steps=0, full_range=180):
    if num_steps == 0:
        return np.array([0]), 1
    else:
        return np.linspace(-full_range/2, full_range/2, num_steps), num_steps


if __name__ == '__main__':
    xx, yy, N = spiral_scan(1, 1, 0.025)
    plt.scatter(xx, yy)
    plt.gca().set_aspect('equal')
    plt.show()
