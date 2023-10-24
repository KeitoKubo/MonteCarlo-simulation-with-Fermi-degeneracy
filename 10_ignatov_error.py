import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy import constants

plt.style.use("scientific")
fig, ax = plt.subplots()

e = constants.e
F = 1e5 # 1 kV/cm
hbar = constants.hbar 
m_star = 0.1 * constants.m_e 
tau_e = 1e-12 
tau_i = 1e-11 
we = 1e12
wi = 1e11
w0 = we + wi
tau_0 = 1 / w0

def dispersion(k):
    return hbar * hbar/2 * k * k/m_star

def ignatov(n):
    t = 0
    k = 0
    Esum = 0
    count = n
    while count > 0:
        rand = random.uniform(0,1.0)
        tf = -tau_0 * math.log(1 - rand)
        t += tf
        k += e * F / hbar * tf
        # ignatov
        if rand < (we / w0):
            if rand > 0.5:
                k = -k
        else:
            k = 0
        Esum += dispersion(k) 
        count -= 1
    v_drift = Esum / (e*F*t)
    return v_drift * 1e-5


# 配列の生成
x_index = np.arange(0.0,6.0,0.05) # シミュレーション回数の指数部
x = 10 ** x_index # シミュレーション回数
y = [] # ドリフト速度の配列
for val_x in x:
    val = int(val_x)
    val_y = ignatov(val)
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

# プロット
ax.plot(x, z)
plt.xscale('log')
plt.yscale('log')

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$z$")
plt.show()