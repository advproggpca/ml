import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Sigmoid activation function where sum of w's and b's is fed
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Generate dataset of values b/w -10 and 10 with interval of .5
X = np.arange(-10, 11, 0.5)
# The actual parameters of w and b
# w = 2
# b = -3
y = (sigmoid(2 * X - 3) >= 0.5).astype(int)


# shuffling data so that training / testing has both +ive and -ive items
np.random.seed(42)
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

X_train, X_test, y_train, y_test = X[:30], X[30:], y[:30], y[30:]

# Start model with random values of w and b
w = np.random.rand()
b = np.random.rand()


# Predict the values of training data based on random weights
Z = w * X_train + b
y_train_pred = (sigmoid(Z) >= 0.5).astype(int)
evl = np.array(y_train_pred == y_train)

right = np.sum(evl == True)
wrong = np.sum(evl == False)

accuracy = right / len(X_train)

recall = np.sum((y_train_pred == 1) & (y_train == 1)) / np.sum(y_train == 1)

precision = np.sum((y_train_pred == 1) & (y_train == 1)) / np.sum(y_train_pred == 1)


# plotting the decision boundary
x_values = np.linspace(-10, 10, 100)
z_values = w * x_values + b
y_values = sigmoid(z_values)
plt.plot(x_values, y_values, label='Sigmoid Function')
plt.plot(x_values, sigmoid(2 * x_values - 3), c="g", label="True Labels")
plt.scatter(X_train, y_train, color='red', label='Training Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()


# plotting roc curve
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.plot(recall, precision, marker='o', label='ROC Curve')
plt.xlabel('Recall')


# Confusion Matrix
cm = confusion_matrix(y_train, y_train_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Loss function
def binary_cross_entropy(y_train, y_train_pred):
    epsilon = 1e-15

    y_train_pred = np.clip(y_train_pred, epsilon, 1 - epsilon)

    bce = -np.mean(
        y_train * np.log(y_train_pred) +
        (1 - y_train) * np.log(1 - y_train_pred)
    )


bce = binary_cross_entropy(y_train, y_train_pred)