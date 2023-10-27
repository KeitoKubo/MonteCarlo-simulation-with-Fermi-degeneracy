import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy import constants

plt.style.use("scientific")
fig, ax = plt.subplots()

#constants
e = constants.e
tau_0 = 5e-11
delta_t = 5e-13
F = [1e5,0.1] 
hbar = constants.hbar
m_star = 0.1 * constants.m_e

e_num = int(1e4) # electron numbers
k = np.zeros((2,e_num))

def dispersion(k_val):
    return hbar * hbar/2 * k_val * k_val/m_star

Esum = 0
cur_time = 0
const_num = e*F[0]/hbar
time_arr = [0]
v_drift_ave = [0]

while cur_time < tau_0 :
    cur_time += delta_t
    time_arr.append(cur_time)
    for i in range(e_num):
        rand = random.uniform(0,1)
        if rand < (delta_t/tau_0):
            Esum -= dispersion(k[0][i])
            k[0][i] = 0
        else :
            k[0][i] += const_num * delta_t
            Esum += dispersion(const_num * delta_t)
    v_drift_val = Esum / (e*F[0]*cur_time*e_num) * 1e-5
    v_drift_ave.append(v_drift_val)


ax.plot(time_arr,v_drift_ave)

ax.set_xlabel(r"time")
ax.set_ylabel(r"Drift velocity$(10^5 m/s)$")

# asymptotic line
true_v_drift = e * tau_0 / m_star
plt.hlines(true_v_drift,0,time_arr[len(time_arr)-1],linestyles='dotted')

# show them
plt.show()