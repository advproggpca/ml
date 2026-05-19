import numpy as np
# Wrapping the vectors in NumPy arrays

input_vector = np.array([1.66, 1.56])

weights_1 = np.array([1.45, -0.66])

bias = np.array([0.0])

def sigmoid(x):
   return 1 / (1 + np.exp(-x))

def make_prediction(input_vector, weights, bias):
    return sigmoid(np.dot(input_vector, weights) + bias)

prediction = make_prediction(input_vector, weights_1, bias)

print(f"The prediction result is: {prediction}")