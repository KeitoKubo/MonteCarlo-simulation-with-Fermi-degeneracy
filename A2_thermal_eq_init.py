import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.constants import e, hbar, m_e
from scipy.constants import k as k_b
from scipy.integrate import quad

m_star = 0.1 * m_e
T = 300
E_F_eV = 0.1
E_F = E_F_eV * e # Fermi energy
# Electron density corresponding to Fermi energy
dos2d = m_star / (math.pi * (hbar**2))
n_E_F = dos2d * k_b * T * math.log(1 + math.exp(E_F / (k_b * T)))
# n_E_F = int(n_E_F)

# Abstention method paramaters
df_upper = m_star / (math.pi * (hbar**2))
E_upper = 3*k_b*T*math.log(10) + E_F + k_b*T*math.log(1+math.exp(-1*E_F/(k_b*T)))

#calculate total and mean energy by integration
def f(E_eV):
    v = E_eV / (1 + np.exp((E_eV - E_F_eV) / (k_b * T / e)))
    return e * v / (n_E_F / dos2d)

E_max = np.amax([E_F, 0]) + 20 * k_b * T
mean_E_eV, err = quad(f, 0, E_max / e)
mean_E = mean_E_eV * e

def rand():
    return np.random.rand()

def energy(k_val):
    return hbar**2 * (k_val[0]**2 + k_val[1]**2) / (2 * m_star * e)

# Using FD distribution
def df(E_val):
    return df_upper/ (1.0 + math.exp((E_val - E_F)/(k_b * T)))

def thd():
    while True:
        E_val = rand() * E_upper
        df_val = rand() * df_upper
        df_true = df(E_val)
        if df_val < df_true:
            return E_val
        
e_num = 10000  # number of electrons

#energy of electrons in thermal equilibrium distribution
E_ini = [] 
for i in range(e_num):
    E_ini.append(thd())