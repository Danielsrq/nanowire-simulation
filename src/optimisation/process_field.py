import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import griddata


def get_meta_data(file_path: str):
    headers = {}
    with open(file_path) as f:
        for line in f:
            if line[0] != '#': break
            if ':' in line:
                key_value = line.split('# ')[1].split(':')
                key, value = key_value[0], key_value[1].split('\n')[0].strip()
                headers[key] = value
    return int(headers['xnodes']), int(headers['ynodes']), int(headers['znodes']), \
           float(headers['xstepsize']), float(headers['ystepsize']), float(headers['zstepsize'])


def interpolate_field(file_path: str, nx: int, ny: int):
    """
    Takes in a stray field ovf file
    For the original (x x y) dimensions for the gridspace, expand the space to
    have nx x ny points and fill in the empty coordinates by interpolation
    """
    x, y, z, _, _, _ = get_meta_data(file_path)
    zslice = int(z / 2)
    data = np.array(np.loadtxt(file_path))
    data_field = data.reshape(x, y, z, 3, order="F")
    u, v = data_field[:, :, :, 0][:, :, zslice], \
        data_field[:, :, :, 1][:, :, zslice]
    x_grid, y_grid = np.mgrid[0:x, 0:y]
    plt.imshow(u.T, extent=(0, x, 0, y), origin='lower')
    plt.show()
    x_points = [i for i in range(x)]
    y_points = [i for i in range(y)]
    points = np.transpose(np.vstack([x_grid.ravel(), y_grid.ravel()]))
    x_grid_new, y_grid_new = np.mgrid[0:x-1:nx, 0:y-1:ny]
    grid_u = griddata(points,
                      u.flatten(),
                      (x_grid_new, y_grid_new),
                      method='cubic')
    grid_v = griddata(points,
                      v.flatten(),
                      (x_grid_new, y_grid_new),
                      method='cubic')
    plt.imshow(grid_u.T, extent=(0, x, 0, y), origin='lower')
    plt.show()
    plt.imshow(grid_v.T, extent=(0, x, 0, y), origin='lower')
    plt.show()
    return grid_u, grid_v


interpolate_field('./data/strayfield_halbach_100_60.ovf', nx=1000j, ny=1000j)
