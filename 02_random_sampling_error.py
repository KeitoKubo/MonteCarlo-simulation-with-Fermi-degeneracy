import numpy as np
import matplotlib.pyplot as plt
import random
import math

plt.style.use("scientific")
fig, ax = plt.subplots()

# 1回のモンテカルロ積分を行う関数
def monte_carlo_cal(n):
    count = 0
    for i in range(n):
        x = random.uniform(0,1.0)
        y = random.uniform(0,1.0)
        if x*x + y*y < 1:
            count += 1
    pi = 4.0 * count / n
    return pi

# 配列の生成
x_index = np.arange(0.0,6.0,0.05)
x = 10 ** x_index
y = []
for val_x in x:
    val = int(val_x)
    val_y = monte_carlo_cal(val)
    val_y = val_y / math.pi
    y.append(val_y)

# 相対誤差の計算
err_base = abs(1-y[0])
z = []
for i in range(len(y)):
    err = abs(1-y[i])/err_base
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