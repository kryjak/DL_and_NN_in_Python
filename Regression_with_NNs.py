"""
For binary classification, we used the logistic sigmoid at the final layer.
For multiclass classification, we used the softmax instead.
If we use a linear transformation, we can use the NN to perform regression, not classification.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

def derivative_W1(X, Z, W2, Y, T):
    # return X.T.dot((T - Y).dot(W2.T) * (1 - Z * Z))
    return X.T.dot((T - Y).dot(W2.T) * (Z > 0))  # RELU

def derivative_b1(Z, W2, Y, T):
    # return np.sum((T - Y).dot(W2.T) * (1 - Z * Z), axis=0)
    return np.sum((T - Y).dot(W2.T) * (Z > 0), axis=0)  # RELU

def derivative_W2(Z, Y, T):
    return Z.T.dot(T - Y)

def derivative_b2(Y, T):
    return np.sum(T - Y, axis=0)

def feedforward(X, W1, b1, W2, b2):
    Z = X.dot(W1) + b1
    Z = Z * (Z > 0)  # Relu
    Y = Z.dot(W2) + b2  # no softmax at the end !!!
    return Z, Y

def cross_entropy(T, pY):  # least-squared errors
    return np.mean((T - pY) ** 2)

N = 500
X = np.random.random((N, 2))*4 - 2  # X points between -2 and 2 in 2D
T = (X[:,0]*X[:,1]).reshape((N, 1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], T)
plt.show()

D = X.shape[1]
M = 100
K = 1

W1 = np.random.randn(D, M) / np.sqrt(D)
b1 = np.zeros(M)
W2 = np.random.randn(M, K) / np.sqrt(M)
# W2 = np.random.randn(M) / np.sqrt(M)
b2 = 0

cost = []
learning_rate = 1e-4
steps = 200

for ii in range(steps):
    Z, pY = feedforward(X, W1, b1, W2, b2)
    crossentropy = cross_entropy(T, pY)
    cost.append(crossentropy)
    if ii % 25 == 0:
        print(crossentropy)

    W1 += learning_rate * derivative_W1(X, Z, W2, pY, T)
    b1 += learning_rate * derivative_b1(Z, W2, pY, T)
    W2 += learning_rate * derivative_W2(Z, pY, T)
    b2 += learning_rate * derivative_b2(pY, T)

plt.plot(cost)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], pY)

# surface plot
line = np.linspace(-2, 2, 20)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
_, pY = feedforward(Xgrid, W1, b1, W2, b2)
pY = pY.reshape(400)
ax.plot_trisurf(Xgrid[:, 0], Xgrid[:, 1], pY, linewidth=0.2, antialiased=True)
plt.show()
