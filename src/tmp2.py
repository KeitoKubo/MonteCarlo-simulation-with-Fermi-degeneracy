'''
(Linux version)
this program generate gifs of time lapse of distribution function.
I considered Fermi degeneracy effect.
'''
import joblib
import time as tm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import rand
from scipy.constants import e, hbar, m_e
from scipy.constants import k as k_b
from scipy.integrate import quad
from matplotlib import animation
from matplotlib import rcParams

E_F_arr = [10e-3, 20e-3, 30e-3, 50e-3]

def EMC(index):
    m_star = 0.1 * m_e  # effective mass (kg)
    T = 10  # temperature (K)
    kT = k_b * T / e  # thermal energy (eV)
    E_F = E_F_arr[index]  # Fermi level (eV)

    F = np.array([0,0])
    F_x = F[0]  # electric field along x (V/m)
    num_e = int(1e5)  # number of electrons
    partition = int(11) # this must be odd number

    E_pho = 60e-3  # phonon energy (eV)
    N_pho = 1 / (np.exp(E_pho / kT) - 1)  # phonon distribution
    g_s = 2 # spin factor

    tau_e = 1e-12  # elastic scattering time (s)
    tau_p = 1e-12  # phonon scattering time for T = 0 (s)

    dos2d = m_star / (np.pi * hbar**2) * e  # density of states (/m2 eV)
    N_e = dos2d * kT * np.log(1 + np.exp(E_F / kT))  # electron density (/m2)

    W_ela = 1 / tau_e  # elastic scattering rate (/s)
    W_emi = (1 + N_pho) / tau_p  # phonon emission rate (/s)
    W_abs = N_pho / tau_p  # phonon absorption rate (/s)
    W_total = W_ela + W_emi + W_abs  # total scattering rate (/s)

    E_max = np.amax([E_F, 0]) + 20 * kT  # cut-off energy (eV)

    def FD(E):
        return 1 / (1 + np.exp((E - E_F) / kT))

    def krandom(ksquared):
        k_abs = np.sqrt(ksquared)
        theta = 2 * np.pi * rand()
        k_x = k_abs * np.cos(theta)
        k_y = k_abs * np.sin(theta)
        return np.array([k_x, k_y])

    def ktoE(k): # k-vector to E scalar
        return hbar**2 * (k[0]**2 + k[1]**2) / (2 * m_star * e)

    def Etok(E): # E scalar to k-vector
        ksquared = 2 * m_star * E * e / hbar**2
        return krandom(ksquared)

    def ene_thd(): # randomly generate a thermal equilibrium energy
        while True:
            E_val = rand() * E_max
            df_val = rand()
            df_true = 1 / (1 + np.exp((E_val - E_F) / kT))
            if df_val < df_true:
                return E_val

    def f(E): # calculate the mean energy
        v = E / (1 + np.exp((E - E_F) / kT))
        return v / (N_e / dos2d)
    E_mean, err = quad(f, 0, E_max)



    ### generate the initial thermal distribution

    k_ini = []
    for i in range(num_e):
        k_ini.append(Etok(ene_thd()))
    k_arr = np.array(k_ini)
    k_arr = k_arr.T # k_arr : k_x, k_y values of each electrons

    k_max = np.max(np.abs(k_arr))
    pow = np.floor(np.log10(k_max))
    k_max = int(np.ceil(k_max / 10**pow)) * 10**pow

    f_k = np.zeros((partition, partition)) # f(k) should be updated periodically
    k_delta = 2 * k_max / partition

    def Get_fk_index(k):
        k_x = k[0] + k_max
        k_y = k[1] + k_max
        x_index = int(k_x / k_delta)
        y_index = int(k_y / k_delta)
        return x_index, y_index


    alpha = (2*np.pi)**2 * N_e / (g_s * num_e * k_delta**2)
    for i in range(len(k_arr[0])):
        x_index, y_index = Get_fk_index(k_arr[:, i])
        f_k[y_index][x_index] += alpha

    # Function which adds f(E) to animation imageset
    T0 = 1e-12  # unit of time
    V0 = 1e5    # unit of drift velocity
    E0 = 1e-3   # unit of energy

    plt.style.use('scientific')
  
    fig = plt.figure(figsize=(12, 6))
    ims = []

    def Addplot():
        f_E_arr = []
        for y_index in range((partition - 1) // 2, partition):
            for x_index in range((partition - 1) // 2, partition):
                kx = (x_index + 1/2) * k_delta - k_max
                ky = (y_index + 1/2) * k_delta - k_max
                E_val = ktoE([kx, ky])
                f_val = f_k[y_index][x_index]
                f_E_arr.append(np.array([E_val,f_val]))
        f_E_arr = np.array(f_E_arr)
        f_E_arr = f_E_arr[np.argsort(f_E_arr[:, 0])]
        img = plt.plot(f_E_arr[:,0] / E0, f_E_arr[:,1], c='k')
        text1 = fig.text(0.45, 0.70, f'${round(cur_time * 1e12, 0)}$ ps',
                ha='left', va='center')
        ims.append(img + [text1])

    E_arr_forF = np.linspace(0,E_max,num=100)
    Fermi_arr = FD(E_arr_forF)

    ### EMC

    sim_time = 50e-12  # simulation time (s)
    delta_t = 4e-14  # time step (s)
    cur_time = 0
    time_arr = []
    v_drift_arr = []
    E_mean_arr = []
    Ei_arr = np.zeros(num_e) # energy array of each electrons
    time_index = 1
    counter = 0 # when this can be diveded by 0, add f to imageset

    # initialize Ei_arr
    for i in range(num_e):
        k_i = k_arr[:,i]
        Ei_arr[i] = ktoE(k_i)
    # main stream
    while cur_time < sim_time:
        counter += 1
        cur_time += delta_t
        if cur_time > time_index* (1e-12):
            print(cur_time)
            time_index += 1
        time_arr.append(cur_time)
        for i in range(num_e):
            k_new = k_arr[:, i] + F * (e * delta_t / hbar) # free flight
            # scattering process
            if rand() < W_total * delta_t:
                r = rand()
                if r < W_ela / W_total:
                    # elastic scattering 
                    k_new = krandom(k_new[0]**2 + k_new[1]**2) 
                elif r < (W_ela + W_emi) / W_total:
                    # phonon emission
                    E_val = ktoE(k_new)
                    if E_val > E_pho:
                        E_val -= E_pho
                        k_new = Etok(E_val)
                else:
                    # phonon absorption
                    E_val = ktoE(k_new)
                    E_val += E_pho
                    k_new = Etok(E_val)
            # check if k could be updated
            x_index_new, y_index_new = Get_fk_index(k_new)
            if x_index_new >= 0 and x_index_new < partition and y_index_new >= 0 and y_index_new < partition:
                if rand() < (1 - f_k[y_index_new][x_index_new]):
                    # updated k
                    x_index, y_index = Get_fk_index(k_arr[:, i])
                    f_k[y_index][x_index] -= alpha
                    f_k[y_index_new][x_index_new] += alpha
                    k_arr[:,i] = k_new
                    Ei_arr[i] = ktoE(k_arr[:, i])
        vd_val = hbar * np.mean(k_arr, axis=1) / m_star
        v_drift_arr.append(vd_val)
        E_mean_arr.append(np.mean(Ei_arr))
        if counter % 10 == 0:
            _ = Addplot()

    time = np.array(time_arr)
    v_drift = np.array(v_drift_arr)
    energy = np.array(E_mean_arr)

    ### Plot Animation
    plt.plot(E_arr_forF / E0, Fermi_arr, c='b')
    fig.text(0.45, 0.40, r'$E_{/rm F}$ = ' + f'${E_F * 1e3}$ meV',
                ha='left', va='center')
    ani = animation.ArtistAnimation(fig, ims, interval = 50)
    ani_name = "EF_" + str(int(E_F * 1e3)) + "meV" + ".gif"
    ani.save('imgs/research_graphs/EMC_degeneracy/v_drift E_mean time response F_0/' + ani_name, writer='pillow')
    plt.show()

_ = joblib.Parallel(n_jobs=-1)(joblib.delayed(EMC)(index) for index in range(len(E_F_arr)))