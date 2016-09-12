"""A basic implementation of linear regression"""

import numpy as np

x = np.array([0.5, 1.5, 2, 2.3, 5.6])
y = np.array([1, 2, 3, 4, 5])

def _mse(a, b):
    """Compute MSE for 2 vectors"""
    return np.mean(np.square(a - b))


def _activation(x, w, b=1):
    """Activation function"""
    return w * x + b


def _calculate_gradient(x, y, w):
    """Calculate MSE gradient for the weights provided"""
    return -2 * np.mean(x * (y - (_activation(x, w))))


def _train_epoch(x, y, w, lr):
    """Train for one epoch using gradient descent"""
    gradient_w = 0
    for i in range(len(x)):
        gradient_w += _calculate_gradient(x[i], y[i], w)
    return w - lr * gradient_w

def linear_regression(x, y, lr=0.01, epochs=100):
    """Calculate linear regression for 1D numpy arrays"""
    w = 1
    b = 1
    for i in range(epochs):
        y_hat = _activation(x, w, b)
        error = _mse(y, y_hat)
        print(error)
        w = _train_epoch(x, y, w, lr)
    print(w)
