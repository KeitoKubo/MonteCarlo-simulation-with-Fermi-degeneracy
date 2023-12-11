import joblib
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from scipy.constants import e, hbar, m_e
from scipy.constants import k as k_b
from scipy.integrate import quad

# T is quite low, simulated T=10
E_F_arr = [10e-3, 20e-3, 30e-3, 50e-3]
k_max_arr = [3e8, 3e8, 4e8, 4e8]

def func(i):
    m_star = 0.1 * m_e  # effective mass (kg)
    T = 10  # temperature (K)
    kT = k_b * T / e  # thermal energy (eV)
    E_F = E_F_arr[i] # Fermi level (eV)

    num_e = int(2e5)  # number of electrons
    partition = int(13) # this must be odd number
    g_s = 2 # spin factor
    k_max = int(k_max_arr[i])

    dos2d = m_star / (np.pi * hbar**2) * e  # density of states (/m2 eV)
    '''N_e = dos2d * kT * np.log(1 + np.exp(E_F / kT))  # electron density (/m2)'''
    E_max = np.amax([E_F, 0]) + 20 * kT  # cut-off energy (eV)
    N_e = dos2d * (E_max - kT * np.log((1+np.exp((E_max-E_F)/kT)) / (1+np.exp((-E_F)/kT))))

    f_k = np.zeros((partition, partition)) # f(k) should be updated periodically
    k_delta = k_max / (partition - 1)

    alpha = (2*np.pi)**2 * N_e / (g_s * num_e * (2 * k_max/partition)**2)

    def krandom(ksquared):
        k_abs = np.sqrt(ksquared)
        theta = 2 * np.pi * rand()
        k_x = k_abs * np.cos(theta)
        k_y = k_abs * np.sin(theta)
        return np.array([k_x, k_y])

    def Etok(E):
        ksquared = 2 * m_star * E * e / hbar**2
        return krandom(ksquared)

    # randomly generate a thermal equilibrium energy
    def ene_thd():
        while True:
            E_val = rand() * E_max
            df_val = rand()
            df_true = 1 / (1 + np.exp((E_val - E_F) / kT))
            if df_val < df_true:
                k_val = Etok(E_val)
                x_index, y_index = get_fk_index(k_val)
                if rand() < 1 - f_k[y_index][x_index]:
                    f_k[y_index][x_index] += alpha
                    return k_val

    def FD(E):
        return 1 / (1 + np.exp((E - E_F) / kT))

    def get_fk_index(k):
        k_x = k[0] + partition * k_delta
        k_y = k[1] + partition * k_delta
        x_index = int(k_x / (2.0 * k_delta))
        y_index = int(k_y / (2.0 * k_delta))
        return x_index, y_index


    ### generate the initial thermal distribution

    k_ini = []
    for i in range(num_e):
        k_ini.append(ene_thd())
    k_arr = np.array(k_ini)
    k_arr = k_arr.T # k_arr : k_x, k_y values of each electrons
    '''
    k_max = -0.1
    for i in range(len(k_arr[0])):
        k_max = max(k_max,max(abs(k_arr[0][i]),abs(k_arr[1][i])))
    #round and get k_max value
    k_max = int(k_max)
    k_max_str = str(k_max)
    k_max_b = int(k_max_str[0]) + 1
    k_max = k_max_b * (10**(len(k_max_str) - 1))
    k_max = float(k_max)
    '''
    


    '''
    for i in range(len(k_arr[0])):
        k_x = k_arr[0][i] + partition * k_delta
        k_y = k_arr[1][i] + partition * k_delta
        x_index = int(k_x / (2.0 * k_delta))
        y_index = int(k_y / (2.0 * k_delta))
        f_k[y_index][x_index] += alpha
    '''
    

    # generate f(E) from f_k

    f_E_arr = []
    FD_E_arr = []

    for y_index in range(int((partition-1)/2), partition, 1):
        for x_index in range(int((partition-1)/2), partition, 1):
            f_val = f_k[y_index][x_index]
            i = y_index - (partition-1)/2
            j = x_index - (partition-1)/2
            E_val = hbar**2 * (2*k_delta)**2 * (i**2 + j**2) / (2 * m_star * e)
            fermi_val = FD(E_val)
            f_E_arr.append(np.array([E_val,f_val]))
            FD_E_arr.append(np.array([E_val,fermi_val]))

    f_E_arr = np.array(f_E_arr)
    FD_E_arr = np.array(FD_E_arr)

    f_E_arr = f_E_arr[np.argsort(f_E_arr[:, 0])]
    FD_E_arr = FD_E_arr[np.argsort(FD_E_arr[:, 0])]

    E0 = 1e-3   # unit of energy
    '''plt.style.use('scientific')'''
    
    fig, ax = plt.subplots()
    ax.text(0.45, 0.40, r'$E_{\rm F}$ = ' + f'${E_F * 1e3}$ meV',
                ha='left', va='center', transform=ax.transAxes)
    ax.plot(f_E_arr[:,0] / E0, f_E_arr[:,1], c='k', label='generated function')
    ax.plot(FD_E_arr[:,0] / E0, FD_E_arr[:,1], c='b', label='Fermi-Dirac function')

    # 軸ラベル
    ax.set_xlabel(r"Energy (meV)")

    plt.legend()
    plt.show()

_ = joblib.Parallel(n_jobs=-1)(joblib.delayed(func)(i) for i in range(len(E_F_arr)))