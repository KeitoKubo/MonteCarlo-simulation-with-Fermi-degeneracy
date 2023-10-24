import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy import constants

plt.style.use("scientific")
fig, ax = plt.subplots()

""""
e = 1
t_max = 10
tau_0 = 1
F = 1
hbar = 1 #暫定
m_star = 1 #暫定
"""
e = constants.e
tau_0 = 1e-12
F = 1e5
hbar = constants.hbar
m_star = 0.1 * constants.m_e

def dispersion(k):
    return hbar * hbar/2 * k * k/m_star

def ezaki_model_func(n):
    t = 0
    k = 0
    Esum = 0
    count = n
    while count > 0:
        tf = -tau_0 * math.log(1 - random.uniform(0,1.0))
        t += tf
        k += e * F / hbar * tf
        Esum += dispersion(k)
        k = 0
        count -= 1
    v_drift = Esum / (e*F*t)
    return v_drift * 1e-5

def synchronous_ensemble_func(n):
    t = 0
    k = 0
    ksum = 0
    N = 0
    count = n
    while count > 0:
        tf = -tau_0 * math.log(1 - random.uniform(0,1.0))
        t += tf
        k += e * F / hbar * tf
        ksum += k
        N += 1
        k = 0
        count -= 1
    v_drift = hbar * (ksum / N) /(m_star)
    return v_drift * 1e-5

# 配列の生成
x1_index = np.arange(0.0,6.0,0.05)
x1 = 10 ** x1_index
y1 = []
for val_x in x1:
    val = int(val_x)
    val_y = ezaki_model_func(val)
    y1.append(val_y)
y2 = []
for val_x in x1:
    val = int(val_x)
    val_y = synchronous_ensemble_func(val)
    y2.append(val_y)

# プロット
ax.plot(x1, y1,label="ezaki model")
ax.plot(x1,y2,label = 'synchronous ensemble method')
plt.xscale('log')

# 軸ラベル
ax.set_xlabel(r"scattering count")
ax.set_ylabel(r"drift velocity $(10^5 m/s)$")

#漸近線
true_v_drift = e * tau_0 / m_star
plt.hlines(true_v_drift,0,x1[len(x1)-1],linestyles='dotted')
ax.text(19.0,0.6,"asymptotic value : $y$ = " + str(format(true_v_drift, '.2f')))

# 確認
plt.legend()
plt.show()