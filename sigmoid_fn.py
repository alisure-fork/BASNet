from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x, a):
    return 1 / (1 + np.exp(-(x - a)))

x = np.arange(0, 20, 0.1)
# sig = sigmoid(x)
plt.plot(x, sigmoid(x, 12))
plt.show()
