# %% Imports
import numpy as np
import os
import errno
from scipy.io import loadmat
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import PillowWriter

plt.style.use(["science", "notebook"])
from scipy import sparse


# %% Animation
my_cmap = plt.get_cmap("cool")


def init(fig, plot):
    # Plot the surface.
    ax.plot_surface(X, Y, plot, cmap=my_cmap, linewidth=0, antialiased=False)
    ax.set_xlabel("$x/a$")
    ax.set_ylabel("$y/a$")
    ax.set_zlabel("$\propto|\psi|^2$")
    return (fig,)


floori = lambda x: int(np.floor(x))
ceili = lambda x: int(np.ceil(x))


def animate(i):
    ax.view_init(elev=10, azim=4 * i)
    return (fig,)


def animation():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", autoscale_on=True)
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=90, interval=50)
    ani.save("rotate_azimuth_angle_3d_surfTEST.gif", writer="pillow", fps=20)


def animate(i, img, data):

    """
    Animation function. Paints each frame. Function for Matplotlib's
    FuncAnimation.
    """

    img.set_data(data[i, ...])  # Fill img with the modulus data of the wave function.
    # eigplot.set_ydata(energies[i, :])
    img.set_zorder(1)

    return (img,)  # We return the result ready to use with blit=True.


def adjust_size(arr, shape):
    shape = np.asarray(shape)
    tmp = np.array(np.ceil(shape / np.asarray(arr.shape)), dtype=int)
    arr = np.kron(arr, np.ones(tmp))
    rem = np.array([arr.shape[0] - shape[0], arr.shape[1] - shape[1]]) / 2
    arr = arr[
        floori(rem[0]) : arr.shape[0] - ceili(rem[0]),
        floori(rem[1]) : arr.shape[1] - ceili(rem[1]),
    ]
    return arr


def mkdir(path):
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise OSError("Creation of the directory %s failed" % path)
