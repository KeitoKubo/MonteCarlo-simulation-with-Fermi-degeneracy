import numpy as np
import joblib
import matplotlib.pyplot as plt
import random
import math
from scipy.constants import e, hbar, m_e
from scipy.constants import k as k_b
from scipy.integrate import quad

E_F_arr = [1e-3,1e-2,1e-1]

def EMC(i):
    
    m_star = 0.1 * m_e
    T = 300
    E_F_eV = E_F_arr[i]
    E_F = E_F_eV * e # Fermi energy
    # Electron density corresponding to Fermi energy
    dos2d = m_star / (math.pi * (hbar**2))
    n_E_F = dos2d * k_b * T * math.log(1 + math.exp(E_F / (k_b * T)))
    # n_E_F = int(n_E_F)

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

    Esaki = False  # Esaki or Ignatov
    F = np.array([1e5, 0e5])  # electric field
    e_num = 10000  # number of electrons
    sim_time = 5e-11  # simulation time
    delta_t = 4e-14  # time step

    if Esaki:
        tau_e = np.inf
        tau_i = 1e-12
    else:
        tau_e = 1e-12
        tau_i = 5e-12

    we = 1 / tau_e
    wi = 1 / tau_i
    w0 = we + wi
    tau_0 = 1 / w0

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

    # EMC
    def inelastic(index):
        E_val = thd()
        k_val = math.sqrt(2*m_star*E_val) / hbar
        r = np.random.rand()
        theta = 2*math.pi*r
        k_x = k_val * math.cos(theta)
        k_y = k_val * math.sin(theta)
        k[:,index] = [k_x,k_y]

    cur_time = 0  # current time
    time = []
    v_drift = []
    Ei = np.zeros(e_num)
    E = []
    while cur_time < sim_time:
        cur_time += delta_t
        time.append(cur_time)
        Esum = np.array([0.0,0.0])
        for i in range(e_num):
            # free flight
            k[:, i] += e * F[:] / hbar * delta_t
            # scattering
            if rand() < (delta_t / tau_0):
                if rand() < (we / w0):
                    # elastic
                    if rand() > 0.5:
                        k[:, i] = -k[:, i]
                else:
                    inelastic(i)
            Ei[i] = energy(k[:, i])
        vd = hbar * np.mean(k, axis=1) / m_star
        v_drift.append(vd)
        E.append(np.mean(Ei))
    time = np.array(time)
    v_drift = np.array(v_drift)
    E = np.array(E)

    # Plotting

    T0 = 1e-12  # unit of time
    V0 = 1e5    # unit of drift velocity
    E0 = 1e-3   # unit of energya


    plt.style.use("scientific")
    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(1, 2, 1)

    ax.plot(time / T0, v_drift[:, 0] / V0, c='k')
    ax.plot(time / T0, v_drift[:, 1] / V0, c='b')

    v_asym = e * tau_0 / m_star * F
    ax.axhline(v_asym[0] / V0, c='k', ls='dotted')
    ax.axhline(v_asym[1] / V0, c='b', ls='dotted')

    ax.set_xlabel(r"Time (ps)")
    ax.set_ylabel(r"Drift Velocity ($10^5$ m/s)")
    ax.text(15,0.5,"$E_F$ = {} eV".format(E_F_eV),fontsize="xx-large")

    ax = fig.add_subplot(1, 2, 2)

    ax.plot(time / T0, E / E0, c='k')

    E_asym = e * (F[0]**2 + F[1]**2) * tau_0/m_star * tau_i  +  (mean_E/e)
    ax.axhline(E_asym / E0, c='k', ls='dotted')
    ax.axhline((mean_E/e) / E0, c='b', ls='dotted')
    print((mean_E/e) / E0)
    n_E_F = int(n_E_F)
    n_E_F = str(n_E_F)
    len_n_E_F = str(len(n_E_F)-1)
    plot_point = ((E_asym + mean_E/e) / 2) / E0
    ax.text(20,plot_point,"$n$={}.{} x 10$^{{{}}}$".format(n_E_F[0], n_E_F[1], len_n_E_F),fontsize="xx-large")

    ax.set_xlabel(r"Time (ps)")
    ax.set_ylabel(r"Mean Energy (meV)")

    fig.tight_layout()

    plt.show()

_ = joblib.Parallel(n_jobs=-1)(joblib.delayed(EMC)(i) for i in range(len(E_F_arr)))