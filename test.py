### EMC which considers Fermi degeneracy
import joblib
import time as tm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import rand
from scipy.constants import e, hbar, m_e
from scipy.constants import k as k_b
from scipy.integrate import quad
from matplotlib import animation
from matplotlib import rcParams

plt.rcParams['font.family'] = ['STIX Two Text']
fig, ax = plt.subplots()
ax.plot([1, 2, 3], label='test')

ax.legend()
plt.show()