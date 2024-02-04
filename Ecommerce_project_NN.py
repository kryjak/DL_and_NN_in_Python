"""
In logistic_softmax_train.py, it is shown how to use logistic regression (i.e. no hidden layers) for multiclass
classification (softmax instead of the logistic function) on the ecommerce project data.
Here, we will apply what we've learned in training_NNs.py to use a simple (1 hidden layer) NN on this data.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from Training_NNs import derivative_W2, derivative_W1, derivative_b2, derivative_b1, feedforward, predict
from process import get_data

Xtrain, Ytrain, Xtest, Ytest = get_data()

Ntrain, D = Xtrain.shape
Ntest, _ = Xtest.shape
M = 5
K = len(set(Ytrain) | set(Ytest))  # the OR covers the case that some classes are not included in the test or train set

Ttrain = np.zeros((Ntrain, K), dtype=int)
Ttest = np.zeros((Ntest, K), dtype=int)

for ii in range(Ntrain):
    Ttrain[ii, Ytrain[ii]] = 1

for ii in range(Ntest):
    Ttest[ii, Ytest[ii]] = 1

# randomly initialize weights
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

cost = []
classification_rate = []
learning_rate = 1e-3
steps = 10000

for ii in range(steps):
    Z, pY = feedforward(Xtrain, W1, b1, W2, b2)
    if ii % 100 == 0:
        preds = predict(pY)
        classification_rate.append( np.mean(preds == Ytrain).round(2) )
        cost.append(np.sum(Ttrain * np.log(pY)))

    W1 += learning_rate * derivative_W1(Xtrain, Z, W2, pY, Ttrain)
    b1 += learning_rate * derivative_b1(Z, W2, pY, Ttrain)
    W2 += learning_rate * derivative_W2(Z, pY, Ttrain)
    b2 += learning_rate * derivative_b2(pY, Ttrain)

plt.plot(cost)
plt.title('Cross-entropy loss in gradient descent')
plt.show()

print(f'Final classification rate for the train set: {classification_rate[-1]}')

_, pYtest = feedforward(Xtest, W1, b1, W2, b2)
print(f'Classification rate for the test set: {np.mean(predict(pYtest) == Ytest).round(2)}')
