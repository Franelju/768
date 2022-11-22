#%%  Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.integrate import simps
import pandas as pd
import os


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


# %%
