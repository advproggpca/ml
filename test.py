import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

print(sigmoid(2 * (1.5) - 3))