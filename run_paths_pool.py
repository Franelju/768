import math
import numpy as np
from timebudget import timebudget
from multiprocessing import Pool
import pandas as pd

from Combined_fxns_single_2D import combine


#%run ../SWIMMERS/Model/Combined-fxns-single-2D.ipynb

# --- param for MSSI fig----
# set a: (DATA_1, but discrepancy with nu?) V = 6, mu =0.01, nu = 1147
# set b: V = 6, mu = 0.05, nu = 230.9
# set c: V = 6, mu = 0.1 , nu = 116.3
# set d: V = 6, mu = 0.25, nu = 47.6
# set e: V = 6, mu = 0.5, nu = 24.7
# set f: V = 6, mu = 1, nu = 13.4
# set g: DATA_10 V = 6, mu = 3, nu = 6.2
# set h: V = 6, mu = 5, nu = 5.0


# --- parameters ----
# set 1: V = 6, mu = 0.01, nu = 1000, tau=4.73 # original
# set 2: V = 6, mu = 0.0075, nu = 1530
# set 3: V = 6, mu = 0.0125, nu = 918
# set 4: V = 4.5, mu = 0.01, nu = 646
# set 5: V = 7.5, mu = 0.01, nu = 1793
# set 6: V = 6.75, mu = 0.01125, nu = 1291
# set 7: V = 5.25, mu = 0.01125, nu = 781
# set 8: V = 6.75, mu = 0.00875, nu = 1659
# set 9: V = 5.25, mu = 0.00875, nu = 1004
# set 10: V = 6, mu = 3, nu = 6.2, tau= 8.58
# set 11: V = 3, mu = 0.01, nu = 287, tau= 4.67
# set 12: V = 3, mu = 3, nu = 2.4,

# ---picking based on tau values from paper contour figure---- (tau between 4 and 10)
# set 13: V = 3, mu= 0.005, nu= 574 , tau=  4.84 #what would have been set 14 in the order is set 11
# set 14: V = 3, mu= 0.05, nu= 58, tau= 4.64
# set 15: V = 3, mu= 0.1, nu= 30, tau= 4.40
# set 16: V = 3, mu=0.5, nu= 6.7, tau= 6.82
# set 17: V = 3, mu=1, nu= 4, tau= 10.44

# set 18: V = 6, mu=0.005, nu= 2014 , tau= 5.08   #what would have been set 19 in the order is set 1
# set 19: V = 6, mu=0.02, nu= 500, tau= 4.22
# set 20: V = 6, mu=0.1, nu= 100, tau= 3.98
# set 21: V = 6, mu=0.5, nu= 30, tau= 4.25
# set 22: V = 6, mu=1, nu= 13.8, tau= 5.52
# set 23: V = 6, mu=2, nu= 7.9   , tau=  6.98  #what would have been set 24 in the order is set 10
# set 24: V = 6, mu=4, nu= 5.5, tau= 10.34

mu = 0.1
phi = 1
nu = 1000
eps = 0.75
tstart = 0
tend = 30  # 1000 # normally this is 30
tstep = tend * 800
savename = "mu={:0d}".format(mu) + str(mu) + ";nu={:0d}".format(nu) + ".csv"
print("Savename is " + savename)
# ---- call model ----
def run_many_paths(S):

    return combine(
        U0=0,
        V0=0,
        tstart=tstart,
        tend=tend,
        tstep=tstep,
        mu=mu,
        phi=phi,
        nu=nu,
        eps=eps,
        seed=S,
    )


# --- seed paths ----
seeds = np.array(
    [
        88029,
        24637,
        184098,
        47818,
        123588,
        167310,
        184315,
        82857,
        54521,
        37794,
        159379,
        105627,
        46542,
        80451,
        18142,
        73880,
        36003,
        97594,
        139500,
        98251,
        79929,
        164572,
        28675,
        61221,
        52185,
        112782,
        140616,
        94253,
        142967,
        49319,
        108751,
        39501,
        21373,
        4986,
        65318,
        196028,
        126378,
        116540,
        85475,
        36670,
        138788,
        101997,
        23320,
        69240,
        60811,
        46097,
        99804,
        94486,
        180624,
        9444,
        98863,
        73341,
        123207,
        76067,
        121428,
        155052,
        82351,
        195020,
        138585,
        105495,
        176691,
        104937,
        123693,
        172677,
        39612,
        133934,
        52600,
        131253,
        71428,
        122009,
        52957,
        145786,
        181730,
        180793,
        11914,
        16778,
        5965,
        152723,
        95198,
        106402,
        151978,
        172675,
        66790,
        120178,
        55152,
        68825,
        13171,
        9854,
        4203,
        90157,
        197747,
        71659,
        184018,
        175728,
        104692,
        181508,
        150692,
        70904,
        197749,
        73927,
    ]
)

# ---- run in parallel -----

worker_num = 24  # how many workers, None means default (max workers)

if __name__ == "__main__":

    inputs = seeds[0:worker_num]

    pool = Pool(worker_num)  # gathers pool of workers

    result = pool.map(
        run_many_paths, inputs
    )  # maps function calls to the workers, by iterating over inputs


# --- compile paths -----
k = 0
tpts = np.linspace(tstart, tend, tstep)

final = pd.DataFrame()

for k in range(0, len(result)):

    df = pd.DataFrame(
        {
            "test": k,
            "mu": mu,
            "nu": nu,
            "phi": phi,
            "eps": eps,
            "t": tpts,
            "X": result[k][0],
            "X_int": np.r_[0, 0, result[k][2]],
            "X_inc": result[k][4],
            "dGx": result[k][8],
            "Y": result[k][1],
            "Y_int": np.r_[0, 0, result[k][3]],
            "Y_inc": result[k][5],
            "dGy": result[k][9],
            "seed": seeds[k],
        }
    )
    k = k + 1
    final = pd.concat([final, df])

# ---- save data ------

savename = "mu={:0d}".format(mu) + str(mu) + ";nu={:0d}".format(nu) + ".csv"
final.to_csv(savename)
print("Simulation finished")
