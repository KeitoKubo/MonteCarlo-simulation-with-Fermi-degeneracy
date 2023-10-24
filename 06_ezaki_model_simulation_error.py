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
y = []
for val_x in x:
    val = int(val_x)
    val_y = ezaki_model_func(val)
    y.append(val_y)

# 相対誤差の計算
true_v_drift = e * tau_0 / m_star
z = []
for i in range(len(y)):
    err = abs(true_v_drift - y[i]) / true_v_drift
    z.append(err)
z_index = []
for val_z in z:
    z_index.append(math.log10(val_z))

# 近似直線
coe = np.polyfit(x_index,z_index,1)
print(coe)
z_index_pred = coe[1] + coe[0] * x_index
ax.text(19.0,0.6,'y =' + str(format(coe[0], '.2f')) + 'x' + '+'+ str(format(coe[1], '.2f')))

# プロット
ax.plot(x, z)
plt.xscale('log')
plt.yscale('log')

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$z$")
plt.plot(10**x_index, 10**z_index_pred)
plt.show()