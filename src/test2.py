import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy.random import rand
from scipy.constants import e, hbar, m_e
from scipy.constants import k as k_b
from scipy.integrate import quad
from variables import os_windows, T, partition, num_e, m_star

num_e = int(1e5)
cell_row = partition
kT = k_b * T / e  # thermal energy (eV)
E_F = 10e-3  # Fermi level (eV)
F = np.array([1e4,0])
F_x = F[0]  # electric field along x (V/m)
sim_time_index = 50

E_pho = 60e-3  # phonon energy (eV)
N_pho = 1 / (np.exp(E_pho / kT) - 1)  # phonon distribution
g_s = 2 # spin factor

dos2d = m_star / (np.pi * hbar**2) * e  # density of states (/m2 eV)
N_e = dos2d * kT * np.log(1 + np.exp(E_F / kT))  # electron density (/m2)

tau_e = 1e-12  # elastic scattering time (s)
tau_p = 1e-12  # phonon scattering time for T = 0 (s)
W_ela = 1 / tau_e  # elastic scattering rate (/s)
W_emi = (1 + N_pho) / tau_p  # phonon emission rate (/s)
W_abs = N_pho / tau_p  # phonon absorption rate (/s)
W_total = W_ela + W_emi + W_abs  # total scattering rate (/s)
tau = 1 / W_total
print("tau: " + str(tau))
k_bar = e*F_x*tau/hbar
E_bar = hbar**2 * k_bar**2 / (2*m_star*e) * 1e3
print("kbar: " + str(k_bar))
print("E_bar: " + str(E_bar) + " meV")