import joblib
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from scipy.constants import e, hbar, m_e
from scipy.constants import k as k_b
from scipy.integrate import quad

E_F_arr = [10e-3, 20e-3, 30e-3, 50e-3]

def func(i):
    m_star = 0.1 * m_e  # effective mass (kg)
    T = 4.2  # temperature (K)
    kT = k_b * T / e  # thermal energy (eV)
    E_F = E_F_arr[i]  # Fermi level (eV)
    os_windows = True

    F = np.array([0,0])
    F_x = F[0]  # electric field along x (V/m)
    num_e = int(1e5)  # number of electrons
    partition = int(31) # this must be odd number

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

    def ktoE(k):
        return hbar**2 * (k[0]**2 + k[1]**2) / (2 * m_star * e)

    def Etok(E):
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
    

    # generate f(E) from f_k, just converting

    f_E_arr = []
    E_arr_forF = np.linspace(0,E_max,num=100)
    FD_E_arr = FD(E_arr_forF)

    for y_index in range((partition - 1) // 2, partition):
        for x_index in range((partition - 1) // 2, partition):
            kx = (x_index + 1/2) * k_delta - k_max
            ky = (y_index + 1/2) * k_delta - k_max
            E_val = ktoE([kx, ky])
            f_val = f_k[y_index][x_index]
            f_E_arr.append(np.array([E_val,f_val]))

    f_E_arr = np.array(f_E_arr)

    f_E_arr = f_E_arr[np.argsort(f_E_arr[:, 0])]

    E0 = 1e-3   # unit of energy
    plt.style.use('scientific')
    if os_windows:
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'STIX Two Text'
        plt.rcParams['font.family'] = ['STIX Two Text']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    if os_windows:
        fig.text(0.55, 0.40, r'$\mathrm{E_{\mathrm{F}}}$ = ' + f'${E_F * 1e3}$ meV',
                ha='left', va='center', transform=ax.transAxes, fontsize = 24)
        fig.text(0.55, 0.50, f'$0$ ps',
                ha='left', va='center', transform=ax.transAxes,fontsize = 24)
    else:
        fig.text(0.55, 0.40, r'$E_{\rm F}$ = ' + f'${E_F * 1e3}$ meV',
                ha='left', va='center', transform=ax.transAxes, fontsize = 24)
        fig.text(0.55, 0.50, f'$0$ ps',
                ha='left', va='center', transform=ax.transAxes,fontsize = 24)
    ax.plot(f_E_arr[:,0] / E0, f_E_arr[:,1], c='k', label='generated function')
    ax.plot(E_arr_forF / E0, FD_E_arr, c='b', label='Fermi-Dirac function')

    ax.set_xlabel(r'Energy (meV)')
    ax.set_ylabel(r'Occupancy rate')

    plt.legend()
    fig.savefig('imgs/EMC_degeneracy/dist_function_F_0/' + "EF_" + str(int(E_F * 1e3)) + "meV" + "_" + str(int(0)))

_ = joblib.Parallel(n_jobs=-1)(joblib.delayed(func)(i) for i in range(len(E_F_arr)))
