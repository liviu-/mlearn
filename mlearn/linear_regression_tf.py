"""Multiple linear regression using TensorFlow"""

import numpy as np
import tensorflow as tf

x = np.array([[i, i + 10] for i in range(100)]).astype(np.float32)
y = np.array([i * 0.4 + j * 0.9 + 1 for i, j in x]).astype(np.float32)

# Add bias
x = np.hstack((x, np.ones((x.shape[0], 1)))).astype(np.float32)

# Create variable for weights
n_features = x.shape[1]
np.random.rand(n_features)
w = tf.Variable(tf.random_normal([n_features, 1]))
w = tf.Print(w, [w])

# Loss function
y_hat = tf.matmul(x, w)
loss = tf.reduce_mean(tf.square(tf.sub(y, y_hat)))

operation = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(loss)

with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    for iteration in range(5000):
        session.run(operation)
    weights = w.eval()
    print(weights)
