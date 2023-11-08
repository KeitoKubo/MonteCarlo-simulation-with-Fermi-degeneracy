import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.constants import e, hbar, m_e

# constants
delta_t = 2e-14
F = 1e5
m_star = 0.1 * m_e
sim_time = 5e-11  # simulation time

tau_e = 1e-12
tau_i = 5e-12
we = 1 / tau_e
wi = 1 / tau_i
w0 = we + wi
tau_0 = 1 / w0

e_num = 10000  # number of electrons
k = np.zeros(e_num)
Ei = np.zeros(e_num)

def dispersion(k_val):
    return hbar**2 * k_val**2 / (2 * m_star)

def rand():
    return np.random.rand()

# EMC
cur_time = 0  # current time
time = []
v_drift = []
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
    vd = hbar * np.mean(k) / m_star
    v_drift.append(vd)
time = np.array(time)
v_drift = np.array(v_drift)

plt.style.use("scientific")
fig, ax = plt.subplots()

T0 = 1e-12  # unit of time
V0 = 1e5    # unit of drift velocity

ax.plot(time / T0, v_drift / V0, c='k')

v_asym = e * tau_0 / m_star * F
ax.axhline(v_asym / V0, c='k', ls='dotted')


ax.set_xlabel(r"Time (ps)")
ax.set_ylabel(r"Drift Velocity ($10^5$ m/s)")

plt.show()
