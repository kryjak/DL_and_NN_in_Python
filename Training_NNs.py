"""
In this notebook, I will reproduce on my own the contents of backprop.py from the course materials.
This is just for practice purposes and to make sure I can implement feed-forward, backpropagation etc.
We're going to use a simple NN with two layers.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
import backprop

# this is the only thing I take from the course materials. The rest will be implemented on my own.
# create the data
Nclass = 500
D = 2  # dimensionality of input
M = 3  # hidden layer size
K = 3  # number of classes

X1 = np.random.randn(Nclass, D) + np.array([0, -2])
X2 = np.random.randn(Nclass, D) + np.array([2, 2])
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])

labels = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)  # labels for the 3 classes
N = len(labels)  # total number of samples

plt.scatter(X[:, 0], X[:, 1], c=labels, s=100, alpha=0.5)
plt.show()

# create the indicator matrix for the targets
T = np.zeros((N, K), dtype=int)
for ii in range(N):
    T[ii, labels[ii]] = 1

def feedforward(X, W1, b1, W2, b2):
    a = X.dot(W1) + b1
    Z = sp.expit(a)
    alpha = Z.dot(W2) + b2
    Y = sp.softmax(alpha, axis=1)
    return Z, Y

# randomly initialize weights
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

cost = []
learning_rate = 1e-3
steps = 1000

for ii in range(steps):
    Z, Y = feedforward(X, W1, b1, W2, b2)
    if ii % 100 == 0:
        preds = np.argmax(Y, axis=1)
        classification_rate = np.mean(preds == np.argmax(T)).round(2)
        cost.append(np.sum(T * np.log(Y)))

    dJdW1 = X.T.dot((T - Y).dot(W2.T) * Z * (1 - Z))
    dJdb1 = np.sum((T - Y).dot(W2.T) * Z * (1 - Z), axis=0)
    dJdW2 = Z.T.dot(T - Y)
    dJdb2 = np.sum(T - Y, axis=0)

    W1 += learning_rate * dJdW1
    b1 += learning_rate * dJdb1
    W2 += learning_rate * dJdW2
    b2 += learning_rate * dJdb2

plt.plot(cost)
plt.title('Cross-entropy loss in gradient descent')
plt.show()
