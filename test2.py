import joblib
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from scipy.constants import e, hbar, m_e
from scipy.constants import k as k_b
from scipy.integrate import quad

m_star = 0.1 * m_e  # effective mass (kg)
F_x = 1e5  # electric field along x (V/m)
num_e = 10000  # number of electrons
delta_t = 4e-15
k = e * F_x / hbar * delta_t
print(hbar**2 * (k**2) / (2 * m_star * e) * 1e3)