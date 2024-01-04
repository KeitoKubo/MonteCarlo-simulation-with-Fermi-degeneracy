#Generate Initial Distribution like getting electrons follow FD distribution.
# N_mc is calculated later in this method.
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy.random import rand
from scipy.constants import e, hbar, m_e
from scipy.constants import k as k_b
from scipy.integrate import quad
from variables import os_windows, T, partition, num_e, m_star

E_F_arr = [5e-3, 10e-3, 15e-3]

def EMC_NoElectricField_gifs(joblib_index):
	num_e = int(1e5)
	cell_row = partition
	kT = k_b * T / e  # thermal energy (eV)
	E_F = E_F_arr[joblib_index]  # Fermi level (eV)
	F = np.array([0,0])
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

	def Get_fk_index(k):
		k_x = k[0] + k_max
		k_y = k[1] + k_max
		x_index = int(k_x / k_delta)
		y_index = int(k_y / k_delta)
		return x_index, y_index


	### generate the initial thermal distribution

	f_arr_k = np.zeros((cell_row, cell_row))
	k_delta = 2 * k_max / cell_row # length of one cell edge
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

	# Function which adds f(E) to animation imageset
	T0 = 1e-12  # unit of time
	V0 = 1e5    # unit of drift velocity
	E0 = 1e-3   # unit of energy

	plt.style.use('scientific')
	if os_windows:
		plt.rcParams['mathtext.fontset'] = 'custom'
		plt.rcParams['mathtext.rm'] = 'STIX Two Text'
		plt.rcParams['font.family'] = ['STIX Two Text']

	fig, ax = plt.subplots(figsize=(12, 6))
	ims = []

	def Addplot():
		f_E_arr = []
		for y_index in range((partition - 1) // 2, partition):
			for x_index in range((partition - 1) // 2, partition):
				kx = (x_index + 1/2) * k_delta - k_max
				ky = (y_index + 1/2) * k_delta - k_max
				E_val = ktoE([kx, ky])
				f_val = f_arr_k[y_index][x_index]
				f_E_arr.append(np.array([E_val,f_val]))
		f_E_arr = np.array(f_E_arr)
		f_E_arr = f_E_arr[np.argsort(f_E_arr[:, 0])]
		if int(round(cur_time * 1e12, 0)) == 0:
			img = plt.plot(f_E_arr[:,0] / E0, f_E_arr[:,1], c='k', label='generated function')
		else:
			img = plt.plot(f_E_arr[:,0] / E0, f_E_arr[:,1], c='k')
		text1 = fig.text(0.7, 0.50, f'${round(cur_time * 1e12, 0)}$ ps',
				ha='left', va='center', fontsize = 24)
		ims.append(img + [text1])

	E_arr_forF = np.linspace(0,E_max,num=100)
	Fermi_arr = FD(E_arr_forF)

	### EMC

	sim_time =  sim_time_index * 1e-12  # simulation time (s)
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
				if rand() < (1 - f_arr_k[y_index_new][x_index_new]):
					# updated k
					x_index, y_index = Get_fk_index(k_arr[:, i])
					f_arr_k[y_index][x_index] -= alpha
					f_arr_k[y_index_new][x_index_new] += alpha
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
	ax.set_xlabel(r'Energy (meV)')
	ax.set_ylabel(r'Occupancy rate')
	plt.plot(E_arr_forF / E0, Fermi_arr, c='b', label='Fermi-Dirac function')
	plt.legend()
	if os_windows:
			fig.text(0.7, 0.40, r'$\mathrm{E_{\mathrm{F}}}$ = ' + f'${E_F * 1e3}$ meV',
				ha='left', va='center', fontsize = 24)
	else:
		fig.text(0.7, 0.40, r'$E_{F}$ = ' + f'${E_F * 1e3}$ meV',
				ha='left', va='center')
	ani = animation.ArtistAnimation(fig, ims, interval = 50)
	ani_name = "EF_" + str(int(E_F * 1e3)) + "meV" + ".gif"
	fig.tight_layout()
	ani.save('../imgs/EMC_degeneracy/EMC_NoElectricField_gifs/' + ani_name, writer='pillow')
	fig.savefig('../imgs/EMC_degeneracy/EMC_NoElectricField_gifs/' + "EF_" + str(int(E_F * 1e3)) + "meV" + "_" + str(int(sim_time_index)))

_ = joblib.Parallel(n_jobs=-1)(joblib.delayed(EMC_NoElectricField_gifs)(index) for index in range(len(E_F_arr)))