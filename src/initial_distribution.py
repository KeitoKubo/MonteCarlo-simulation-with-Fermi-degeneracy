#Generate Initial Distribution which get electrons follow FD distribution.
# N_mc is calculated later in this method.
import joblib
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from scipy.constants import e, hbar, m_e
from scipy.constants import k as k_b
from scipy.integrate import quad
from variables import os_windows, T, partition, num_e, m_star

E_F_arr = [5e-3,10e-3,15e-3]

def init_dist_func(joblib_index):
	num_e = int(1e5)
	cell_row = partition
	kT = k_b * T / e  # thermal energy (eV)
	E_F = E_F_arr[joblib_index]  # Fermi level (eV)

	E_pho = 60e-3  # phonon energy (eV)
	N_pho = 1 / (np.exp(E_pho / kT) - 1)  # phonon distribution
	g_s = 2 # spin factor

	dos2d = m_star / (np.pi * hbar**2) * e  # density of states (/m2 eV)
	N_e = dos2d * kT * np.log(1 + np.exp(E_F / kT))  # electron density (/m2)

	E_max = np.amax([E_F, 0]) + 20 * kT  # cut-off energy (eV)
	k_max = np.sqrt(2 * m_star * E_max * e / hbar**2)
	pow = np.floor(np.log10(k_max))
	k_max = int(np.ceil(k_max / 10**pow)) * 10**pow


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

	def f(E): # calculate the mean energy
		v = E / (1 + np.exp((E - E_F) / kT))
		return v / (N_e / dos2d)
	E_mean, err = quad(f, 0, E_max)


	### generate the initial thermal distribution

	f_arr_k = np.zeros((cell_row, cell_row))
	k_delta = 2 * k_max / cell_row # length of one cell edge
	alpha = (2*np.pi)**2 * N_e / (g_s * num_e * k_delta**2)
	fd_arr_k = np.zeros((cell_row, cell_row))
	p_num_arr = np.zeros((cell_row, cell_row))

	def index_to_k(y_index, x_index):
		k_x = x_index * k_delta - k_max + 1/2 * k_delta
		k_y = y_index * k_delta - k_max + 1/2 * k_delta
		k_mid = np.array([k_x, k_y])
		return k_mid

	#calculate raw FD val
	for y_index in range(cell_row):
		for x_index in range(cell_row):
			k_mid = index_to_k(y_index, x_index)
			E_mid = ktoE(k_mid)
			fd_val = FD(E_mid)
			fd_arr_k[y_index][x_index] = fd_val

	#convert them into particle numbers
	fd_arr_sum = np.sum(fd_arr_k)
	for y_index in range(cell_row):
		for x_index in range(cell_row):
			p_num_arr[y_index][x_index] = int(fd_arr_k[y_index][x_index] * num_e / fd_arr_sum)
	p_num_arr = p_num_arr.astype(int)
	num_e = np.sum(p_num_arr)
	alpha = (2*np.pi)**2 * N_e / (g_s * num_e * k_delta**2)
	for y_index in range(cell_row):
		for x_index in range(cell_row):
			f_arr_k[y_index][x_index] = alpha * p_num_arr[y_index][x_index]

	#generate initial k_arr
	k_arr = []
	for y_index in range(cell_row):
		for x_index in range(cell_row):
			k_mid = index_to_k(y_index, x_index)
			for i in range(p_num_arr[y_index][x_index]):
				k_arr.append(k_mid)
	k_arr = np.array(k_arr)
	k_arr = k_arr.T

	# generate f(E) from f_k, just converting
	f_E_arr = []
	E_arr_forF = np.linspace(0,E_max,num=100)
	FD_E_arr = FD(E_arr_forF)

	for y_index in range(cell_row):
		for x_index in range(cell_row):
			E_mid = ktoE(index_to_k(y_index, x_index))
			f_E_arr.append(np.array([E_mid, f_arr_k[y_index][x_index]]))

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
	fig.tight_layout()
	fig.savefig('../imgs/EMC_degeneracy/initial_distribution/' + "EF_" + str(int(E_F * 1e3)) + "meV" + "_" + str(int(0)))

_ = joblib.Parallel(n_jobs=-1)(joblib.delayed(init_dist_func)(i) for i in range(len(E_F_arr)))