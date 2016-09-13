# mlearn

Sort of a playground to implement things and learn TF.


## Installation

```
$ python3 -m pip install "git+https://github.com/liviu-/mlearn"
```

## Usage

```python
>>> import numpy as np
>>> from mlearn import LinearRegression
>>> # Create a simple dataset 
>>> X_train = np.array([[i] for i in range(100)])
>>> y = np.array([i for i in range(100)])
>>> X_test = np.array([[500], [1000]])
>>> # Create, fit, and predict
>>> lr = LinearRegression()
>>> lr.fit(X_train, y, lr=0.000001, epochs=1000)
Error: 0.0007409217751234417
Weights: [0.991854, 0.05403126]
>>> lr.predict(X_test)
array([ 499.64672907, 999.2394269])
```
