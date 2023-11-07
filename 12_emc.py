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
tau_0 = 1 / we
delta_t = 1e-14
F = np.array([1e5, 0e5])
m_star = 0.1 * m_e
sim_time = 5e-11  # simulation time

e_num = 10000  # number of electrons
k = np.zeros((2, e_num))

# EMC
cur_time = 0  # current time
time = []
v_drift = []
while cur_time < sim_time:
    cur_time += delta_t
    time.append(cur_time)
    for i in range(e_num):
        # free flight
        k[:, i] += e * F[:] / hbar * delta_t
        # scattering
        rand = random.uniform(0, 1)
        if rand < (delta_t / tau_0):
            k[:, i] = 0
    vd = hbar * np.mean(k, axis=1) / m_star
    v_drift.append(vd)
time = np.array(time)
v_drift = np.array(v_drift)

plt.style.use("scientific")
fig, ax = plt.subplots()

T0 = 1e-12  # unit of time
V0 = 1e5    # unit of drift velocity

ax.plot(time / T0, v_drift[:, 0] / V0, c='k')
ax.plot(time / T0, v_drift[:, 1] / V0, c='b')

# asymptotic line
v_asym = e * tau_0 / m_star * F
ax.axhline(v_asym[0] / V0, c='k', ls='dotted')
ax.axhline(v_asym[1] / V0, c='b', ls='dotted')

ax.set_xlabel(r"Time (ps)")
ax.set_ylabel(r"Drift Velocity ($10^5$ m/s)")

plt.show()
