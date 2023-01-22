#%%  Imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.integrate import simps
from aux_funs import mkdir
import pandas as pd
import os


# Animation imports
import imageio as iio
import moviepy.editor as mp

mpl.use("Agg")

#%% define functions
def func(Ut, Us, Vt, Vs, t, s, mu, phi, nu):

    p = (np.pi) / 2
    exponential = (mu * phi * nu * p) * np.exp(
        -((Ut - Us) ** 2 + (Vt - Vs) ** 2) / (4 * (1 + mu * (t - s)))
    )
    poly = (1 + mu * (t - s)) ** (-2)

    return exponential * poly


vecfun = np.vectorize(func, excluded=["t", "Ut", "Vt", "mu", "phi", "nu"])


def combine(U0, V0, tstart, tend, tstep, mu, phi, nu, eps, seed):
    """
    this function does something
    """

    tpts = np.linspace(0, tend, tstep)
    dt = (tend - tstart) / (tstep - 1)
    print("dt=", dt)

    # generate 2*tstep random numbers and reshape to be (tstep,2)
    # generate all noise
    if seed == "on":
        np.random.seed(2222222)
        dG = np.random.normal(0, np.sqrt(dt), tstep * 2)
    elif seed == "off":
        np.random.seed(None)
        dG = np.random.normal(0, np.sqrt(dt), tstep * 2)
    elif type(seed) == int:
        np.random.seed(seed)
        dG = np.random.normal(0, np.sqrt(dt), tstep * 2)

    # reshape into dGx and dGy
    dGx = []
    dGy = []
    for i in range(0, len(dG)):
        if (i % 2) == 0:  # if i is even
            dGx.append(dG[i])
        if (i % 2) != 0:  # if i is odd
            dGy.append(dG[i])

    # initialize
    U = []
    V = []
    U.append(U0)
    U.append(U0 + eps * dGx[0] * dt)
    V.append(V0)
    V.append(V0 + eps * dGy[0] * dt)
    Uinc = []
    Vinc = []
    U_integral = []
    V_integral = []
    Uinc.append(0)
    Uinc.append(eps * dGx[0] * dt)
    Vinc.append(0)
    Vinc.append(eps * dGy[0] * dt)

    for j in range(
        1, tstep - 1
    ):  # this is the "final" time as we creep across the segment

        prefactorU = U[0 : (j + 1)] - np.array(U[j])
        func_vals_x = np.multiply(
            vecfun(
                U[j],
                U[0 : (j + 1)],
                V[j],
                V[0 : (j + 1)],
                tpts[j],
                tpts[0 : (j + 1)],
                mu,
                phi,
                nu,
            ),
            -prefactorU,
        )

        prefactorV = V[0 : (j + 1)] - np.array(V[j])
        func_vals_y = np.multiply(
            vecfun(
                V[j],
                V[0 : (j + 1)],
                U[j],
                U[0 : (j + 1)],
                tpts[j],
                tpts[0 : (j + 1)],
                mu,
                phi,
                nu,
            ),
            -prefactorV,
        )

        U_int = (
            scipy.integrate.simps(func_vals_x, tpts[0 : (j + 1)], axis=-1, even="avg")
            * dt
        )
        V_int = (
            scipy.integrate.simps(func_vals_y, tpts[0 : (j + 1)], axis=-1, even="avg")
            * dt
        )

        # compute next step
        Umove = U_int + eps * dGx[j]
        Vmove = V_int + eps * dGy[j]  # was np.trapz

        # store deterministic integral values
        U_integral.append(U_int)
        V_integral.append(V_int)

        # store total step (integral + noise)
        Uinc.append(Umove)
        Vinc.append(Vmove)

        # compute next position (current position + step)
        U.append(U[j] + Umove)  # = U[j+1]
        V.append(V[j] + Vmove)  # = V[j+1]

    Ufinal = U[-1]
    Vfinal = V[-1]

    return (U, V, U_integral, V_integral, Uinc, Vinc, Ufinal, Vfinal, dGx, dGy)


# %%#%% Parameters
mu = 1000  # 0.01
phi = 1
nu = 1  # 1147.6
eps = 0.75
tstart = 0
tend = 1  # normally this is 30 #1000
Nt = tend * 800
parts = 10
U0, V0 = 0, 0
seed = "off"
#%% Simulate
data = combine(U0, V0, tstart, tend, Nt, mu, phi, nu, eps, seed)
# %%
U, V = np.array(data[0]), np.array(data[1])
plt.plot(U, V)
xmax, xmin, ymax, ymin = U.max(), U.min(), V.max(), V.min()
scale_factor = 1.0
xmax, xmin, ymax, ymin = (
    xmax * scale_factor,
    xmin / scale_factor,
    ymax * scale_factor,
    ymin / scale_factor,
)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.grid()
plt.show()

# %%

particles = 9
Y = np.zeros((Nt, 2, particles))
for i in np.arange(particles):
    data = combine(U0, V0, tstart, tend, Nt, mu, phi, nu, eps, seed)
    Y[:, 0, i], Y[:, 1, i] = np.array(data[0]), np.array(data[1])
    plt.plot(Y[:, 0, i], Y[:, 1, i])
xmax, xmin, ymax, ymin = (
    Y[:, 0, :].max(),
    Y[:, 0, :].min(),
    Y[:, 1, :].max(),
    Y[:, 1, :].min(),
)
scale_factor = 1.25
xmax, xmin, ymax, ymin = (
    xmax * scale_factor,
    xmin * scale_factor,
    ymax * scale_factor,
    ymin * scale_factor,
)
#%%
for i in np.arange(particles):
    plt.plot(Y[:, 0, i], Y[:, 1, i])

xmax, xmin, ymax, ymin = (
    Y[:, 0, :].max(),
    Y[:, 0, :].min(),
    Y[:, 1, :].max(),
    Y[:, 1, :].min(),
)
scale_factor = 1.05
xmax, xmin, ymax, ymin = (
    xmax * scale_factor,
    xmin * scale_factor,
    ymax * scale_factor,
    ymin * scale_factor,
)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.grid()
plt.show()
# %% Animation
print("Processing")
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
frames = []
title = "Swimmers with $\\mu$={:.0e} and $\\nu$={:.0e}".format(mu, nu)
for t in range(Nt):
    for i in np.arange(particles):
        ax.plot(Y[0:t, 0, i], Y[0:t, 1, i])
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid()
    fig.canvas.draw_idle()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image)
    ax.cla()
# %% Save video
savename = "mu=" + str(mu) + "_nu=" + str(nu)
print("Savename is " + savename)
mkdir("animation")
savefolder = "animation/" + savename
iio.mimwrite(savefolder + ".mp4", frames, format="FFMPEG", fps=120)
# mp.VideoFileClip(savename + ".gif").write_videofile(
#     savename + ".mp4", verbose=False, logger=None
# )
# os.remove(savename + ".gif")
print(savename + "video done")

# %%
