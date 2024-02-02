"""
In this notebook, I will reproduce on my own the contents of backprop.py from the course materials.
This is just for practice purposes and to make sure I can implement feed-forward, backpropagation etc.
We're going to use a simple NN with two layers.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

# X - input matrix, size NxD
# Z - hidden layer, size NxM
# Y - output layer, size NxK
# T - target one-hot encoded as an indicator matrix, size NxK
# W2 - weight matrix of the second layer
def derivative_W1(X, Z, W2, Y, T):
    return X.T.dot((T - Y).dot(W2.T) * Z * (1 - Z))
def derivative_b1(Z, W2, Y, T):
    return np.sum((T - Y).dot(W2.T) * Z * (1 - Z), axis=0)
def derivative_W2(Z, Y, T):
    return Z.T.dot(T - Y)
def derivative_b2(Y, T):
    return np.sum(T - Y, axis=0)

def feedforward(X, W1, b1, W2, b2):
    a = X.dot(W1) + b1
    Z = sp.expit(a)
    alpha = Z.dot(W2) + b2
    Y = sp.softmax(alpha, axis=1)
    return Z, Y

def predict(pY):
    return np.argmax(pY, axis=1)

def main():
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
            preds = predict(Y)
            classification_rate = np.mean(preds == labels).round(2)
            cost.append(np.sum(T * np.log(Y)))

        W1 += learning_rate * derivative_W1(X, Z, W2, Y, T)
        b1 += learning_rate * derivative_b1(Z, W2, Y, T)
        W2 += learning_rate * derivative_W2(Z, Y, T)
        b2 += learning_rate * derivative_b2(Y, T)

    plt.plot(cost)
    plt.title('Cross-entropy loss in gradient descent')
    plt.show()
    print(f'Final classification rate: {classification_rate}')

if __name__ == '__main__':
    main()
