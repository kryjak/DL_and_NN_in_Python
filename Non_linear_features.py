"""
In the logistic regression course, we saw that we can deal with non-linear features (XOR and donut problems)
only if we manually engineer extra features into the model.
On the other hand, NNs can automatically learn such non-linear features.
Of course this comes at a cost - more hyperparameters, choice of the activation function, number of hidden layers,
number of hidden units within each such layer.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

# we can't quite reuse the files from Training_NNs.py, because here we are going to use tanh as the activation function
# for the hidden layer, followed by expit (binary classification) for the output layer.

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
    a = X.dot(W1) + b1
    # Z = sp.expit(a)
    Z = X.dot(W1) + b1
    Z = Z * (Z > 0)
    alpha = Z.dot(W2) + b2
    Y = sp.expit(alpha)
    return Z, Y

def predict(pY):
    # return np.argmax(pY, axis=1)
    return np.round(pY)  # expit returns only one number, i.e. P(Y=1|X)

def cross_entropy(T, pY):
    return -np.mean(T * np.log(pY) + (1 - T) * np.log(1 - pY))

# Important note: we can't use the cross-entropy function from multi-class NNs, because there the targets are
# one-hot encoded, here it's just a vector.
# For example, imagine that the indicator matrix is one-hot encoded as [[0, 1], [1, 0]]. Then, T*log(pY) will cover
# both the first (class 1) and second (class 0) rows.
# On the other hand, if we supply just a list, [1, 0], then the second value will be missed. So we need to use
# the binary classification cross-entropy, T*log(Y) + (1-T)*log(1-Y)


def test_XOR():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # input
    T = np.array([0, 1, 1, 0]).reshape((4, 1))  # targets

    D = X.shape[1]
    M = 5
    K = 1  # only one class for binary classification (i.e. P(0) = 1-P(1))

    # randomly initialize weights
    W1 = np.random.randn(D, M)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K)
    b2 = 0

    cost = []
    classification_rate = []
    learning_rate = 1e-2
    steps = 30000

    for ii in range(steps):
        Z, pY = feedforward(X, W1, b1, W2, b2)
        preds = predict(pY)
        classification_rate.append(np.mean(preds == T).round(2))
        crossentropy = cross_entropy(T, pY)
        if cost and crossentropy > cost[-1]:
            print('Early exit')
            break
        cost.append(crossentropy)

        W1 += learning_rate * derivative_W1(X, Z, W2, pY, T)
        b1 += learning_rate * derivative_b1(Z, W2, pY, T)
        W2 += learning_rate * derivative_W2(Z, pY, T)
        b2 += learning_rate * derivative_b2(pY, T)

    plt.plot(cost)
    plt.title('XOR problem - cross-entropy loss in gradient descent')
    print(f'Final classification rate: {classification_rate[-1]}')
    plt.show()

def test_donut():
    N = 1000
    R_inner = 5
    R_outer = 10

    R1 = np.random.randn(N // 2) + R_inner
    theta_inner = 2 * np.pi * np.random.random(N // 2)
    X_inner = np.concatenate([[R1 * np.cos(theta_inner)], [R1 * np.sin(theta_inner)]]).T

    R2 = np.random.randn(N // 2) + R_outer
    theta_outer = 2 * np.pi * np.random.random(N // 2)
    X_outer = np.concatenate([[R2 * np.cos(theta_outer)], [R2 * np.sin(theta_outer)]]).T

    X = np.concatenate([X_inner, X_outer])
    T = np.array([0] * (N // 2) + [1] * (N // 2)).reshape((N, 1))

    D = X.shape[1]
    M = 8
    K = 1  # only one class for binary classification (i.e. P(0) = 1-P(1))

    # randomly initialize weights
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    cost = []
    classification_rate = []
    learning_rate = 0.00005
    regularisation = 0.2
    steps = 3000

    LL = [] # keep track of log-likelihoods
    learning_rate = 0.0001
    regularization = 0.2

    for ii in range(steps):
        Z, pY = feedforward(X, W1, b1, W2, b2)
        preds = predict(pY)
        rate = np.mean(preds == T).round(2)
        classification_rate.append(rate)
        crossentropy = cross_entropy(T, pY)
        cost.append(crossentropy)

        W1 += learning_rate * (derivative_W1(X, Z, W2, pY, T) - regularisation * W1)
        b1 += learning_rate * (derivative_b1(Z, W2, pY, T) - regularisation * b1)
        W2 += learning_rate * (derivative_W2(Z, pY, T) - regularisation * W2)
        b2 += learning_rate * (derivative_b2(pY, T) - regularisation * b2)

        if ii % 100 == 0:
            print("i:", ii, "classification rate:", rate)

    plt.plot(cost)
    plt.title('Donut problem - cross-entropy loss in gradient descent')
    print(f'Final classification rate: {classification_rate[-1]}')
    plt.show()

if __name__ == "__main__":
    # test_XOR()
    test_donut()
