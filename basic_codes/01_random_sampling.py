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

# プロット
ax.plot(x, y)
plt.xscale('log')

# 軸ラベル
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
 
# 確認
plt.show()