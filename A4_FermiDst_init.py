### EMC which considers Fermi degeneracy
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
E_F = 30e-3  # Fermi level (eV)

F_x = 1e5  # electric field along x (V/m)
num_e = int(1e5)  # number of electrons
partition = int(19) # this must be odd number

E_pho = 60e-3  # phonon energy (eV)
N_pho = 1 / (np.exp(E_pho / kT) - 1)  # phonon distribution
g_s = 2 # spin factor

tau_e = 1e-12  # elastic scattering time (s)
tau_p = 1e-12  # phonon scattering time for T = 0 (s)

sim_time = 50e-12  # simulation time (s)
delta_t = 4e-15  # time step (s)

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


### generate the initial thermal distribution

k_ini = []
for i in range(num_e):
	k_ini.append(Etok(ene_thd()))
k_arr = np.array(k_ini)
k_arr = k_arr.T # k_arr : k_x, k_y values of each electrons

k_max = -0.1
for i in range(len(k_arr[0])):
	k_max = max(k_max,max(abs(k_arr[0][i]),abs(k_arr[1][i])))
#round and get k_max value
k_max = int(k_max)
k_max_str = str(k_max)
k_max_b = int(k_max_str[0]) + 1
k_max = k_max_b * (10**(len(k_max_str) - 1))
k_max = float(k_max)

f_k = np.zeros((partition, partition)) # f(k) should be updated periodically
k_delta = k_max / (partition - 1)

alpha = (2*np.pi)**2 * N_e / (g_s * num_e * (2 * k_max/partition)**2)
for i in range(len(k_arr[0])):
	k_x = k_arr[0][i] + partition * k_delta
	k_y = k_arr[1][i] + partition * k_delta
	x_index = int(k_x / (2.0 * k_delta))
	y_index = int(k_y / (2.0 * k_delta))
	f_k[y_index][x_index] += alpha


k_begin = -1 * (partition - 1) * k_delta
k_end = (1 * (partition - 1) + 1) * k_delta
X,Y = np.mgrid[k_begin:k_end:2*k_delta, k_begin:k_end:2*k_delta]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, f_k, cmap="autumn_r", lw=0.5, rstride=1, cstride=1)
ax.text2D(0.8, 0.9, "$E_F = {}$ meV".format(str(E_F * 1e3)), transform=ax.transAxes)

plt.show()
