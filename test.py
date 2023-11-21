import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.constants import e, hbar, m_e
from scipy.constants import k as k_b
import scipy.integrate as spi

m_star = 0.1 * m_e
T = 300
E_F_eV = 0.1
E_F = E_F_eV * e # Fermi energy
# Electron density corresponding to Fermi energy
n_E_F = m_star * k_b * T / (math.pi * (hbar**2)) * math.log(1 + math.exp(E_F/(k_b*T)))
n_E_F = int(n_E_F)

#estimate error
num = 5

# Abstention method paramaters
df_upper = m_star / (math.pi * (hbar**2))
E_upper = 3*k_b*T*math.log(10) + E_F + k_b*T*math.log(1+math.exp(-1*E_F/(k_b*T)))

#calculate total and mean energy by integration
y = lambda x: x  * np.exp((E_F - x)/(k_b * T)) / (1 + np.exp((E_F - x)/(k_b * T)))
total_E, total_E_err = spi.quad(y, 0,E_upper)
z = lambda x: np.exp((E_F - x)/(k_b * T)) / (1 + np.exp((E_F - x)/(k_b * T)))
total_n, total_n_err = spi.quad(z, 0, E_upper)
print(total_E, total_n)
mean_E = total_E / total_n
print(mean_E)