### EMC which considers Fermi degeneracy
import joblib
import time as tm
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from scipy.constants import e, hbar, m_e
from scipy.constants import k as k_b
from scipy.integrate import quad

E_F_arr = [10e-3, 20e-3, 30e-3, 50e-3]
k_max_arr = [3e8, 3e8, 4e8, 4e8]
F_arr = [1e5, 1e5, 1e5, 1e5]

def EMC(index):
	m_star = 0.1 * m_e  # effective mass (kg)
	T = 10  # temperature (K)
	kT = k_b * T / e  # thermal energy (eV)
	E_F = E_F_arr[index]  # Fermi level (eV)

	F = np.array([0,0])
	F[0] = F_arr[index]
	F_x = F[0]  # electric field along x (V/m)
	num_e = int(2e5)  # number of electrons
	partition = int(13) # this must be odd number
	k_max = int(k_max_arr[index])

	E_pho = 60e-3  # phonon energy (eV)
	N_pho = 1 / (np.exp(E_pho / kT) - 1)  # phonon distribution
	g_s = 2 # spin factor

	tau_e = 1e-12  # elastic scattering time (s)
	tau_p = 1e-12  # phonon scattering time for T = 0 (s)

	dos2d = m_star / (np.pi * hbar**2) * e  # density of states (/m2 eV)

	W_ela = 1 / tau_e  # elastic scattering rate (/s)
	W_emi = (1 + N_pho) / tau_p  # phonon emission rate (/s)
	W_abs = N_pho / tau_p  # phonon absorption rate (/s)
	W_total = W_ela + W_emi + W_abs  # total scattering rate (/s)

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
				k_val = Etok(E_val)
				x_index, y_index = Get_fk_index(k_val)
				if rand() < 1 - f_k[y_index][x_index]:
					f_k[y_index][x_index] += alpha
					return k_val

	def f(E): # calculate the mean energy
		v = E / (1 + np.exp((E - E_F) / kT))
		return v / (N_e / dos2d)
	E_mean, err = quad(f, 0, E_max)

	def Get_fk_index(k):
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

	### EMC

	sim_time = 50e-12  # simulation time (s)
	delta_t = 1e-14  # time step (s)
	cur_time = 0
	time_arr = []
	v_drift_arr = []
	E_mean_arr = []
	Ei_arr = np.zeros(num_e) # energy array of each electrons

	# main stream
	while cur_time < sim_time:
		cur_time += delta_t
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
		vd_val = vd = hbar * np.mean(k_arr, axis=1) / m_star
		v_drift_arr.append(vd_val)
		E_mean_arr.append(np.mean(Ei_arr))

	time = np.array(time_arr)
	v_drift = np.array(v_drift_arr)
	energy = np.array(E_mean_arr)

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

begin_time = tm.time()
_ = joblib.Parallel(n_jobs=-1)(joblib.delayed(EMC)(i) for i in range(len(E_F_arr)))
end_time = tm.time()
print(end_time - begin_time)