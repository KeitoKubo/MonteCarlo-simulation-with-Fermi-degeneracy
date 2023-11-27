import joblib
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.random import rand
from scipy.constants import e, hbar, m_e
from scipy.constants import k as k_b
from scipy.integrate import quad

plt.style.use("scientific")

T = 300
E_F = 100 #meV

def b(E_val_eV):
    return math.exp((E_F - E_val_eV) * (0.001) / (k_b * T / e))

def f(E_val_eV):
    return 1.0 / (1.0 + math.exp((E_val_eV - E_F) * (0.001) / (k_b * T / e)))

x = np.linspace(0,150,num=100,dtype='float64')
y = []
z = []
for i in range(len(x)):
    y.append(f(x[i]))
    z.append(b(x[i]))
y = np.array(y)
z = np.array(z)

plt.plot(x,y,color="red",label="Fermi-Dirac")
plt.plot(x,z,color="blue",label="Boltzmann")
plt.xlabel("Energy (meV)")
plt.legend(bbox_to_anchor = (0.5 ,0.5))
plt.show()