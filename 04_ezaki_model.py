import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy import constants

plt.style.use("scientific")
fig, ax = plt.subplots()

t_array = [0]
k_array = [0]

e = constants.e #C
t_max = 1e-11 #s
tau_0 = 1e-12 #s
F = 1e3 #kV/cm
hbar =  constants.hbar # J x s

t = 0
k = 0
while t < t_max:
    tf = -tau_0 * math.log(1 - random.uniform(0,1.0))
    t += tf
    k += e * F / hbar * tf
    t_array.append(t * 1e12)
    k_array.append(k * 1e-6) # 自由走行の終わり
    k = 0
    t_array.append(t * 1e12)
    k_array.append(k * 1e-6) # 散乱直後

# プロット
ax.plot(t_array, k_array)

# 軸ラベル
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
 
# 確認
plt.show()