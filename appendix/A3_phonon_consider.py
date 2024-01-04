import joblib
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from scipy.constants import e, hbar, m_e
from scipy.constants import k as k_b
from scipy.integrate import quad

m_star = 0.1 * m_e  # effective mass (kg)
T = 300  # temperature (K)
kT = k_b * T / e  # thermal energy (eV)
E_F = 60e-3  # Fermi level (eV)

F_x = 1e5  # electric field along x (V/m)
num_e = 10000  # number of electrons

E_pho = 60e-3  # phonon energy (eV)

tau_e = 1e-12  # elastic scattering time (s)
tau_p = 1e-12  # phonon scattering time for T = 0 (s)

sim_time = 50e-12  # simulation time (s)
delta_t = 4e-15  # time step (s)

dos2d = m_star / (np.pi * hbar**2) * e  # density of states (/m2 eV)

E_V_arr = np.linspace(0.0,120.0,100)
E_V_arr *= 1e-3

E_F = 1000
N_e = 1000
def f(E):
  v = 1 / (1 + np.exp((E - E_F) / kT))
  return v / (N_e / dos2d)

y = []
for i in range(len(E_V_arr)):
  E_F = E_V_arr[i]
  N_pho = 1 / (np.exp(E_pho / kT) - 1)  # phonon distribution
  N_e = dos2d * kT * np.log(1 + np.exp(E_F / kT))  # electron density (/m2)
  W_ela = 1 / tau_e  # elastic scattering rate (/s)
  W_emi = (1 + N_pho) / tau_p  # phonon emission rate (/s)
  W_abs = N_pho / tau_p  # phonon absorption rate (/s)
  W_total = W_ela + W_emi + W_abs  # total scattering rate (/s)
  E_max = np.amax([E_F, 0]) + 200 * kT  # cut-off energy (eV)
  N_A, err_a = quad(f, 0, E_pho)
  N_B, err_b = quad(f, E_pho, E_max)
  y_val = ((N_A + N_B) * W_abs - N_B * W_emi) * delta_t * N_pho * 1e3
  y.append(y_val)

y = np.array(y)
E_V_arr *= 1e3

plt.plot(E_V_arr, y, c='black')
plt.xlabel("Fermi energy (eV)")
plt.ylabel("Energy to gain (eV)")
plt.show()