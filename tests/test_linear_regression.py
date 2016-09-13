import numpy as np
import pytest
from mlearn import LinearRegression

@pytest.fixture
def data():
    X_train = np.array([[i] for i in range(100)])
    y = np.array([i for i in range(100)])
    return X_train, y

def test_lr_weights_shape(data):
    X_train, y = data
    lr = LinearRegression()
    lr.fit(X_train, y, lr=0.000001, epochs=500)
    assert lr.w.shape == (2,)
    
def test_lr_prediction_shape(data):
    X_train, y = data
    lr = LinearRegression()
    lr.fit(X_train, y, lr=0.000001, epochs=500)
    prediction = lr.predict(np.array([[200]]))
    assert prediction.shape == (1,)
