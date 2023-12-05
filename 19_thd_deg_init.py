import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import copy
import joblib
from scipy.constants import e, hbar, m_e
from scipy.constants import k as k_b
from scipy.integrate import quad

E_F_arr = [100e-3, 50e-3, 10e-3, 0, -10e-3, -50e-3, -100e-3]

def thd_init(i):
    m_star = 0.1 * m_e
    T = 300
    E_F_eV = E_F_arr[i]
    E_F = E_F_eV * e # Fermi energy
    # Electron density corresponding to Fermi energy
    dos2d = m_star  / (math.pi * (hbar**2))
    n_E_F = dos2d * k_b * T * math.log(1 + math.exp(E_F / (k_b * T)))

    # Abstention method paramaters
    df_upper = m_star / (math.pi * (hbar**2))
    E_upper = 3*k_b*T*math.log(10) + E_F + k_b*T*math.log(1+math.exp(-1*E_F/(k_b*T)))

    #calculate total and mean energy by integration
    def f(E_eV):
        v = E_eV / (1 + np.exp((E_eV - E_F_eV) / (k_b * T / e)))
        return e * v / (n_E_F / dos2d)

    E_max = np.amax([E_F, 0]) + 20 * k_b * T
    mean_E_eV, err = quad(f, 0, E_max / e)
    mean_E = mean_E_eV * e

    def rand():
        return np.random.rand()

    def energy(k_val):
        return hbar**2 * (k_val[0]**2 + k_val[1]**2) / (2 * m_star * e)

    # Using FD distribution
    def df(E_val):
        return df_upper/ (1.0 + math.exp((E_val - E_F)/(k_b * T)))

    def thd():
        while True:
            E_val = rand() * E_upper
            df_val = rand() * df_upper
            df_true = df(E_val)
            if df_val < df_true:
                return E_val
            
    def Etok(E_val):
        return math.sqrt(2*m_star*E_val) / hbar

    e_num = 100000  # number of electrons

    #energy of electrons in thermal equilibrium distribution
    E_ini = [] 
    for i in range(e_num):
        E_ini.append(thd())

    #calculate wave numbers from energy
    k_sca = []
    for i in range(len(E_ini)):
        k_sca.append(Etok(E_ini[i]))

    #calculate wave-number vectors from k_sca
    k = []
    for i in range(len(k_sca)):
        r = np.random.rand()
        theta = 2*math.pi*r
        k_x = k_sca[i] * math.cos(theta)
        k_y = k_sca[i] * math.sin(theta)
        k.append(np.array([k_x, k_y]))
    k = np.array(k)
    k = k.T

    #initial k-space condition

    k_max = -0.1
    for i in range(len(k[0])):
        k_max = max(k_max,max(abs(k[0][i]),abs(k[1][i])))
    #round and get k_max value
    k_max = int(k_max)
    k_max_str = str(k_max)
    k_max_b = int(k_max_str[0]) + 1
    k_max = k_max_b * (10**(len(k_max_str) - 1))
    k_max = float(k_max)
    print(k_max)

    partition = int(159) # this must be odd number

    k_space_particles = np.zeros((partition, partition))
    k_delta = k_max / (partition - 1)
    k_biased = np.copy(k)
    for i in range(len(k_biased[0])):
        k_biased[0][i] += partition * k_delta
        k_biased[1][i] += partition * k_delta

    for i in range(len(k_biased[0])):
        k_x = k_biased[0][i]
        k_y = k_biased[1][i]
        x_index = int(k_x / (2.0 * k_delta))
        y_index = int(k_y / (2.0 * k_delta))
        k_space_particles[y_index][x_index] += 1

    k_space_particles_seq = []
    for i in range(partition):
        for j in range(partition):
            k_space_particles_seq.append(k_space_particles[i][j])
    k_space_particles_seq = np.array(k_space_particles_seq)
    
    k_x = []
    k_y = []
    k_x_val = -1 * (partition - 1) * k_delta
    k_y_val = -1 * (partition - 1) * k_delta

    for i in range(partition):
        k_x.append(k_x_val)
        k_y.append(k_y_val)
        k_x_val += 2 * k_delta
        k_y_val += 2 * k_delta
    k_x = np.array(k_x)
    k_y = np.array(k_y)

    k_begin = -1 * (partition - 1) * k_delta
    k_end = (1 * (partition - 1) + 1) * k_delta
    X,Y = np.mgrid[k_begin:k_end:2*k_delta, k_begin:k_end:2*k_delta]
    X *= k_delta
    Y *= k_delta
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, k_space_particles, cmap="autumn_r", lw=0.5, rstride=1, cstride=1)
    ax.text2D(0.8, 0.9, "$E_F = {}$ meV".format(str(E_F_eV * 1e3)), transform=ax.transAxes)
    #ax.contour(k_x, k_y, k_space_particles, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=-1)
    #ax.contour(k_x, k_y, k_space_particles, 10, lw=3, colors="k", linestyles="solid")

    plt.show()

_ = joblib.Parallel(n_jobs=-1)(joblib.delayed(thd_init)(i) for i in range(len(E_F_arr)))