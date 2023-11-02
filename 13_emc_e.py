import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.constants import e, hbar, m_e

# constants
tau_0 = 1e-11
delta_t = 1e-13
F = np.array([1e5, 0e5])
m_star = 0.1 * m_e
sim_time = 2e-10  # simulation time

e_num = 10000  # number of electrons
k = np.zeros((2, e_num))

# EMC
cur_time = 0  # current time
time = []
E = []
while cur_time < sim_time:
    cur_time += delta_t
    time.append(cur_time)
    Esum = np.array([0.0,0.0])
    for i in range(e_num):
        # free flight
        k[:, i] += e * F[:] / hbar * delta_t
        # scattering
        rand = random.uniform(0, 1)
        if rand < (delta_t / tau_0):
            k[:, i] = 0
        Esum += (hbar**2) * (k[:, i]**2) / (2*m_star)
    Emean = Esum / e_num
    Emean /= e
    E.append(Emean)
time = np.array(time)
E = np.array(E)

plt.style.use("scientific")
fig, ax = plt.subplots()

T0 = 1e-12  # unit of time
V0 = 1e-3    # unit of energy

ax.plot(time / T0, E[:, 0] / V0, c='k')
ax.plot(time / T0, E[:, 1] / V0, c='b')

'''''''''
# asymptotic line
v_asym = e * tau_0 / m_star * F
ax.axhline(v_asym[0] / V0, c='k', ls='dotted')
ax.axhline(v_asym[1] / V0, c='b', ls='dotted')
'''

ax.set_xlabel(r"Time (ps)")
ax.set_ylabel(r"Mean Energy (meV)")

plt.show()
