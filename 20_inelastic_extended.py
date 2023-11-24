import joblib
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from scipy.constants import e, hbar, m_e
from scipy.constants import k as k_b
from scipy.integrate import quad


E_F_arr = [100e-3, 50e-3, 10e-3, 0, -10e-3, -50e-3, -100e-3]

def EMC(i):
    m_star = 0.1 * m_e  # effective mass (kg)
    T = 300  # temperature (K)
    kT = k_b * T / e  # thermal energy (eV)
    E_F = E_F_arr[i]  # Fermi level (eV)

    F_x = 1e5  # electric field along x (V/m)
    num_e = 10000  # number of electrons

    E_pho = 60e-3  # phonon energy (eV)
    N_pho = 1 / (np.exp(E_pho / kT) - 1)  # phonon distribution

    tau_e = 1e-12  # elastic scattering time (s)
    tau_p = 1e-12  # phonon scattering time for T = 0 (s)

    sim_time = 50e-12  # simulation time (s)
    delta_t = 1e-15  # time step (s)

    dos2d = m_star / (np.pi * hbar**2) * e  # density of states (/m2 eV)
    N_e = dos2d * kT * np.log(1 + np.exp(E_F / kT))  # electron density (/m2)

    W_ela = 1 / tau_e  # elastic scattering rate (/s)
    W_emi = (1 + N_pho) / tau_p  # phonon emission rate (/s)
    W_abs = N_pho / tau_p  # phonon absorption rate (/s)
    W_total = W_ela + W_emi + W_abs  # total scattering rate (/s)

    E_max = np.amax([E_F, 0]) + 20 * kT  # cut-off energy (eV)


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


    # randomly generate a thermal equilibrium energy
    def ene_thd():
        while True:
            E_val = rand() * E_max
            df_val = rand()
            df_true = 1 / (1 + np.exp((E_val - E_F) / kT))
            if df_val < df_true:
                return E_val


    # calculate the mean energy
    def f(E):
        v = E / (1 + np.exp((E - E_F) / kT))
        return v / (N_e / dos2d)
    E_mean, err = quad(f, 0, E_max)


    # generate the initial thermal distribution
    k_ini = []
    for i in range(num_e):
        k_ini.append(Etok(ene_thd()))
    k = np.array(k_ini)
    k = k.T


    ### EMC

    print(f'E_F = {E_F * 1e3} meV')
    print(f'N_e = {N_e / 1e4:.2e} /cm2')
    print(f'tau_ela = {1e12 / W_ela:.2f} ps')
    print(f'tau_emi = {1e12 / W_emi:.2f} ps')
    print(f'tau_abs = {1e12 / W_abs:.2f} ps')

    cur_time = 0  # current time
    time = []
    v_drift = []
    energy = []
    Ei = np.zeros(num_e)

    # main stream
    while cur_time < sim_time:
        cur_time += delta_t
        time.append(cur_time)
        for i in range(num_e):
            # free flight
            k[0, i] += e * F_x / hbar * delta_t
            # scattering
            if rand() < W_total * delta_t:
                r = rand()
                if r < W_ela / W_total:
                    # elastic scattering
                    k[:, i] = krandom(k[0, i]**2 + k[1, i]**2)
                elif r < (W_ela + W_emi) / W_total:
                    # phonon emission
                    E = ktoE(k[:, i])
                    if (E > E_pho):
                        E -= E_pho
                        k[:, i] = Etok(E)
                else:
                    # phonon absorption
                    E = ktoE(k[:, i])
                    E += E_pho
                    k[:, i] = Etok(E)
            Ei[i] = ktoE(k[:, i])
        vd = hbar * np.mean(k, axis=1) / m_star
        v_drift.append(vd)
        energy.append(np.mean(Ei))
    time = np.array(time)
    v_drift = np.array(v_drift)
    energy = np.array(energy)


    ### Plotting

    T0 = 1e-12  # unit of time
    V0 = 1e5    # unit of drift velocity
    E0 = 1e-3   # unit of energy

    plt.style.use('scientific')
    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(1, 2, 1)

    ax.set_xlim(0, sim_time / T0)
    ax.plot(time / T0, v_drift[:, 0] / V0, c='k')
    ax.plot(time / T0, v_drift[:, 1] / V0, c='b')

    ax.set_xlabel(r'Time (ps)')
    ax.set_ylabel(r'Drift Velocity ($10^5$ m/s)')

    ax.text(0.45, 0.40, r'$E_{\rm F}$ = ' + f'${E_F * 1e3}$ meV',
            ha='left', va='center', transform=ax.transAxes)
    ax.text(0.45, 0.35, r'$n_{\rm e}$ = ' + f'{N_e / 1e16:.2f} ' + 
            r'$\times 10^{12}\ {\rm cm}^{-2}$',
            ha='left', va='center', transform=ax.transAxes)

    for tau_0 in [1 / (W_ela + W_abs), 1 / (W_ela + W_abs + W_emi)]:
        v_0 = e * tau_0 / m_star * F_x
        ax.axhline(v_0 / V0, c='k', ls=':')

    ax = fig.add_subplot(1, 2, 2)
    ax.set_xlim(0, sim_time / T0)

    ax.plot(time / T0, energy / E0, c='k')
    ax.axhline(E_mean / E0, ls=':')

    ax.set_xlabel(r'Time (ps)')
    ax.set_ylabel(r'Mean Energy (meV)')

    fig.tight_layout()

    plt.show()

_ = joblib.Parallel(n_jobs=-1)(joblib.delayed(EMC)(i) for i in range(len(E_F_arr)))
