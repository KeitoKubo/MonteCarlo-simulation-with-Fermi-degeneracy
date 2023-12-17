import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.constants import e, hbar, m_e

def rand():
    return np.random.rand()

def energy(k):
    return hbar**2 * (k[0]**2 + k[1]**2) / (2 * m_star * e)

Esaki = True  # Esaki or Ignatov
F = np.array([1e5, 0e5])  # electric field
e_num = 10000  # number of electrons
sim_time = 5e-11  # simulation time
delta_t = 1e-14  # time step
m_star = 0.1 * m_e  # effectiv mass

if Esaki:
    tau_e = np.inf
    tau_i = 1e-12
else:
    tau_e = 1e-12
    tau_i = 5e-12

we = 1 / tau_e
wi = 1 / tau_i
w0 = we + wi
tau_0 = 1 / w0

k = np.zeros((2, e_num))
Ei = np.zeros(e_num)

# EMC
cur_time = 0  # current time
time = []
v_drift = []
E = []
while cur_time < sim_time:
    cur_time += delta_t
    time.append(cur_time)
    Esum = np.array([0.0,0.0])
    for i in range(e_num):
        # free flight
        k[:, i] += e * F[:] / hbar * delta_t
        # scattering
        if rand() < (delta_t / tau_0):
            if rand() < (we / w0):
                # elastic
                if rand() > 0.5:
                    k[:, i] = -k[:, i]
            else:
                # inelastic
                k[:, i] = 0
        Ei[i] = energy(k[:, i])
    vd = hbar * np.mean(k, axis=1) / m_star
    v_drift.append(vd)
    E.append(np.mean(Ei))
time = np.array(time)
v_drift = np.array(v_drift)
E = np.array(E)

# Plotting

T0 = 1e-12  # unit of time
V0 = 1e5    # unit of drift velocity
E0 = 1e-3   # unit of energy

plt.style.use("scientific")
fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 2, 1)

ax.plot(time / T0, v_drift[:, 0] / V0, c='k')
ax.plot(time / T0, v_drift[:, 1] / V0, c='b')

v_asym = e * tau_0 / m_star * F
ax.axhline(v_asym[0] / V0, c='k', ls='dotted')
ax.axhline(v_asym[1] / V0, c='b', ls='dotted')

ax.set_xlabel(r"Time (ps)")
ax.set_ylabel(r"Drift Velocity ($10^5$ m/s)")

ax = fig.add_subplot(1, 2, 2)

ax.plot(time / T0, E / E0, c='k')

E_asym = e * tau_i * F * v_asym
ax.axhline(v_asym[0] / V0, c='k', ls='dotted')
ax.axhline(v_asym[1] / V0, c='b', ls='dotted')

ax.set_xlabel(r"Time (ps)")
ax.set_ylabel(r"Mean Energy (meV)")

fig.tight_layout()

plt.show()

