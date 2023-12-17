import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy import constants

plt.style.use("scientific")
fig, ax = plt.subplots()

e = constants.e
tau_0 = 1e-12
F = 1e5
hbar = constants.hbar
m_star = 0.1 * constants.m_e

e_num = 10

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


# 配列の生成
x_index = np.arange(0.0,6.0,0.05)
x = 10 ** x_index
y = [] # 平均をとる

for val_x in x:
    val_sum = 0.0
    for i in range(e_num):
        val = int(val_x)
        val_sum += ezaki_model_func(val)
    val_mean = val_sum / e_num
    y.append(val_mean)


# プロットs
ax.plot(x, y)
plt.xscale('log')

# 軸ラベル
ax.set_xlabel(r"Scattering Count")
ax.set_ylabel(r"Relative Error")
 
#漸近線
true_v_drift = e * tau_0 / m_star
plt.hlines(true_v_drift,0,x[len(x)-1],linestyles='dotted')
# 確認
plt.show()