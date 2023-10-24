import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import constants

e = constants.e
F = 1e5 # 1 kV/cm
hbar = constants.hbar 
m_star = 0.1 * constants.m_e 
we = 1e12
wi = 1e11
w0 = we + wi
tau_0 = 1 / w0

def dispersion(k):
    return hbar**2 * k**2 / (2 * m_star)


def rand():
    return np.random.rand()


def ignatov(n):
    t = 0
    k = 0
    Esum = 0
    count = n
    while count > 0:
        E0 = dispersion(k)
        tf = -tau_0 * math.log(1 - rand())
        t += tf
        k += e * F / hbar * tf
        E1 = dispersion(k)
        # ignatov
        if rand() < (we / w0):
            if rand() > 0.5:
                k = -k
        else:
            k = 0
        Esum += E1 - E0
        count -= 1
    v_drift = Esum / (e * F * t)
    return v_drift

# ドリフト速度の計算
x = np.array(np.logspace(1, 6, 100), dtype=int) # シミュレーション回数
y = np.array([ignatov(val_x) for val_x in x]) # ドリフト速度の配列

plt.style.use("scientific")
fig, ax = plt.subplots()

# プロット
ax.plot(x, y * 1e-5)
ax.set_xscale('log')

# 漸近線
true_v_drift = e * tau_0 / m_star * F
ax.axhline(true_v_drift * 1e-5, ls='dotted')

# 軸ラベル
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

# 確認
plt.show()