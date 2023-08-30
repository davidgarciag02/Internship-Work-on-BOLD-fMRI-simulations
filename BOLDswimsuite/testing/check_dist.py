import numpy as np
import matplotlib.pyplot as plt

B0 = 3
dchi = 3.2989e-8

uniform_distribution = np.linspace(0, 1, 100_000)
theta = np.arccos(2*uniform_distribution - 1)

dBz = B0*4/6*np.pi*dchi*(3*np.cos(theta)**2-1)
dBz_uT = dBz*1e6

plt.hist(dBz_uT, 100)
plt.ylabel('counts')
plt.xlabel('dBz IV (uT)')
plt.show()