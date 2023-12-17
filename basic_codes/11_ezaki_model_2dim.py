import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy import constants

plt.style.use("scientific")
fig, ax = plt.subplots()

# constants
e = constants.e
tau_0 = 1e-12
F = [1e5,0] #Electric Field works only in x-direction
hbar = constants.hbar
m_star = 0.1 * constants.m_e

def dispersion(k):
    return (hbar**2) * (k**2) / (2*m_star)  

# calculate x-direction drift velocity
def ezaki_model_func(n):
    t = 0
    k = 0
    Esum = 0
    count = n
    while count > 0:
        tf = -tau_0 * math.log(1 - random.uniform(0,1.0))
        t += tf
        k += e * F[0] / hbar * tf
        Esum += dispersion(k)
        k = 0
        count -= 1
    v_drift = Esum / (e*F[0]*t)
    return v_drift * 1e-5


# array
scnt_index = np.arange(0.0,6.0,0.05)
scnt = 10 ** scnt_index # scattering count
v_drift = [[],[]]
for val_raw in scnt:
    val = int(val_raw)
    v_drift_val = ezaki_model_func(val)
    v_drift[0].append(v_drift_val)
    v_drift[1].append(0) # does not drift in y-direction

# plot
ax.plot(scnt, v_drift[0],label='x-direction drift velocity')
ax.plot(scnt, v_drift[1],label='y-direction drift velocity')
plt.xscale('log')

ax.set_xlabel(r"Scattering count")
ax.set_ylabel(r"Drift velocity$(10^5 m/s)$")
 
# asymptotic line
true_v_drift = e * tau_0 / m_star
plt.hlines(true_v_drift,0,scnt[len(scnt)-1],linestyles='dotted')

# show them
plt.legend()
plt.show()