# mnist.py

"""
Import the MNIST data and read it
"""
# Using some prebuilt functions to import the data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

"""
Implementing Softmax regression
"""
import tensorflow as tf
# 784 represents each picture (28 x 28 pixels) flattened into a vector
x = tf.placeholder(tf.float32, [None, 784])
# 10 represents each different 'class' the input could be
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

"""
Training the model
"""
# Using cross entropy to detremine the loss
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# Minimizing the loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

"""
Execution
"""
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
	# Instead of using all the data for training, which would be computationally expensive
	# we are using stochastic gradient descent (using small batches of random data)
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

"""
Validation
"""
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
