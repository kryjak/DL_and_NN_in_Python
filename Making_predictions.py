import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

X = np.array([0, 1])
W = np.array([[1, 1], [1, 1]])
V = np.array([1, 1])
b = c = 0

Z = sp.expit(W.T.dot(X) + b)
print(f'First layer predictions are: {Z}')

Ypred = sp.expit(V.dot(Z) + c)
print(f'Output layer prediction is: {Ypred}')

# in general, X will be an NxD matrix of data, rather than just a vector for one sample
# Z will be an NxM matrix, where M - number of nodes in the hidden layer
# so the weight matrix W is DxM, first bias term is 1xM
# then the second set of weights is a vector Mx1 and the second bias term is a scalar
# for binary classification, the output is Nx1 (one prediction for each input data row)
# but for multiple classification, it will be NxK, where K - number of classes

X = np.array([[0, 3.5], [1, 2], [1, 0.5]])  # input data N=3, D=2
Y = np.array([1, 1, 0])  # targets

# Let's do binary classification first
# we will use one hidden layer with M=3 hidden units
# we will also use the tanh activation function

W = np.array([[0.5, 0.1, -0.3], [0.7, -0.3, 0.2]])  # choose random weights
b = np.array([0.4, 0.1, 0])  # bias terms for the first layer
V = np.array([0.8, 0.1, -0.1])  # weights for hidden layer -> output (strictly speaking, this should be a column vector)
c = 0.2  # scalar bias term for hidden layer -> output

print('-'.center(50, '-'))
Z = np.tanh(X.dot(W) + b)
print(f'First layer predictions are: {Z.round(3)}')
Ypred = np.tanh(Z.dot(V) + c)  # strictly speaking it should be V.T, but numpy understand this 
print(f'Output layer prediction is: {Ypred.round(3)}')

"""
When we start to use the sigmoid, the output will be matrix of size NxK, not a vector anymore
But the targets were given as a 1D vector! How do we reconcile this?
Targets in a softmax network are more conveniently expressed as an 'indicator matrix'.
This is similar to `one-hot encoding'.
For example, if we have N=8 and there are K=6 categories to predict, we need to encode the targets in an
8x6 indicator matrix:

targets = np.array([0,5,1,3,1,4,2,0])
target_indicator = np.zeros((8,6), dtype=int)
# a simple for loop to perform the one-hot encoding of targets
for ii in range(len(targets)):
    target_indicator[ii, targets[ii]] = 1

print(onehot)
"""
target_indicator = np.zeros((len(Y), 2), dtype=int)
for ii in range(len(Y)):
    target_indicator[ii, Y[ii]] = 1

print(target_indicator)

# Now let's say that our NN returns the following output:
output = np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
print(f'The classification rate is '
      f'{np.mean([np.argmax(target_indicator, axis=1) == np.argmax(output, axis=1)]).round(2)}.')

""" Let's now get back to the previous example and use softmax with 2 prediction classes."""
V = np.array([[0, 0.8], [0.4, 0.5], [0.5, 0.4]])  # weights for hidden layer -> output
c = np.array([0.3, 0.5])
print(Z.dot(V) + c)
Ypred = sp.softmax(Z.dot(V) + c, axis=1)
print(f'Output for the two-layer prediction is: {Ypred.round(2)}')

print(f'The classification rate is '
      f'{np.mean([np.argmax(target_indicator, axis=1) == np.argmax(Ypred, axis=1)]).round(2)}.')

print('-'.center(50, '-'))
# we're going to generate 3 Gaussian clouds, representing 3 classes
Nclass = 500  # samples per class
# two-dimensional clouds centered at different points
X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])
# targets
Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)

plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
plt.show()

D = 2  # number of samples
M = 3  # number of units in the hidden layer
K = 3  # number of classes

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

def forward(X, W1, b1, W2, b2):
    Z = sp.expit(X.dot(W1) + b1)
    Y = sp.softmax(Z.dot(W2) + b2, axis=1)
    return Y

# a stupid way to get the classification rate
# def classification_rate(Y, P):
#    n_correct = 0
#    n_total = 0
#    for i in range(len(Y)):
#        n_total += 1
#        if Y[i] == P[i]:
#            n_correct += 1
#    return n_correct / n_total

# a much better way
def classification_rate(Y, P):
    return np.mean(Y == P)

P_Y_given_X = forward(X, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis=1)


print(f'Classification rate for randomly chosen weights: {classification_rate(Y, P)}.')
