import tensorflow as tf

# create weight function
def weight(shape, name):
    initial = tf.truncated_normal(shape, stddev=1e-1, dtype=tf.float32)
    return tf.Variable(initial, name=name)

# create bias function
def bias(shape, num, name):
    if num == 0.0: # conv-layer : initialie to 0.0
        initial = tf.zeros(shape, dtype=tf.float32)
    else: # fully-connected layer : initialize to 1.0
        initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

# conv2d wrapping function
def conv(x, y):
    return tf.nn.conv2d(x, y, strides=[1,1,1,1], padding="SAME")