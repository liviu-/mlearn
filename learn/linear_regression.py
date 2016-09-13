"""A basic implementation of linear regression"""
import numpy as np

class LinearRegression():
    """Basic implementation of linear regression"""

    def fit(self, x, y, lr=0.01, epochs=100):
        """Calculate linear regression for numpy ndarrays
        
        Rows should be data points, and columns should be features
        """
        # Add bias
        x = _append_bias(x)
        # Extract number of features
        n_features = x.shape[1]
        w = _initialise_weights(n_features)
        for i in range(epochs):
            error = _mse(y, _activation(x, w))
            w = _train_epoch(x, y, w, lr)
        print("Error: {}".format(error))
        print("Weights: {}".format(w))
        self.w = w

    def predict(self, x):
        """Predict based on the weights computed previously"""
        x = _append_bias(x)
        return _activation(x, self.w)


def _mse(a, b):
    """Compute MSE for 2 vectors"""
    return np.mean(np.square(a - b))


def _activation(x, w):
    """Activation function (dot product in this case)"""
    return np.dot(x, w)


def _partial_derivative_mse(x, y, w):
    """Calculate partial derivatives for MSE"""
    new_w = []
    for feature in x:
        new_w.append(-2 * np.mean(feature * (y - (_activation(x, w)))))
    return new_w


def _train_epoch(x, y, w, lr):
    """Train for one epoch using gradient descent"""
    gradient_w = np.zeros(x.shape[1])
    for i in range(len(x)):
        gradient_w += _partial_derivative_mse(x[i], y[i], w)
    return w - lr * gradient_w


def _initialise_weights(n):
    return np.random.rand(n)

def _append_bias(x):
    """Append 1 to each data point"""
    return np.hstack((x, np.ones((x.shape[0], 1))))
