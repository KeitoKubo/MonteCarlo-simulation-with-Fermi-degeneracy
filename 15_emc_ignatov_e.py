import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.constants import e, hbar, m_e

# constants
tau_e = 1e-12
tau_i = 5e-12
we = 1 / tau_e
wi = 1 / tau_i
w0 = we + wi
tau_0 = 1 / w0
delta_t = 2e-14
F = 1e5
m_star = 0.1 * m_e
sim_time = 5e-11  # simulation time

e_num = 10000  # number of electrons
k = np.zeros(e_num)
Ei = np.zeros(e_num)

def rand():
    return np.random.rand()

# EMC
cur_time = 0  # current time
time = []
E = []
while cur_time < sim_time:
    cur_time += delta_t
    time.append(cur_time)
    for i in range(e_num):
        # free flight
        k[i] += e * F / hbar * delta_t
        # scattering
        rnd = random.uniform(0, 1)
        if rnd < (delta_t / tau_0):
            if rand() < (we / w0):
                if rand() > 0.5:
                    k[i] = -1 * k[i]
            else:
                k[i] = 0
        Ei[i] = hbar**2 * k[i]**2 / (2 * m_star * e)
    E.append(np.mean(Ei))
time = np.array(time)
E = np.array(E)

plt.style.use("scientific")
fig, ax = plt.subplots()

T0 = 1e-12  # unit of time
E0 = 1e-3    # unit of energy

ax.plot(time / T0, E / E0, c='k')

'''
v_asym = e * tau_0 / m_star * F
E_asym = tau_i * F * v_asym # remember unit trans : J - eV
print(E_asym)
ax.axhline(E_asym / E0, c='k', ls='dotted')
'''


ax.set_xlabel(r"Time (ps)")
ax.set_ylabel(r"Mean Energy (meV)")

plt.show()
