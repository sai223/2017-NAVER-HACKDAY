import tensorflow as tf
import time as t
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"  # graphic card number to use

# data csv files
train_csv_dir = "/mnt/hdd3t/Data/hci1/hoon/LightHouse_of_Inha/CSVs/3th/size/train_G_size.csv"
test_csv_dir = "/mnt/hdd3t/Data/hci1/hoon/LightHouse_of_Inha/CSVs/3th/size/test_G_size.csv"

image_height = 299
image_width = 299
train_batch_size = 32 # batch size
test_batch_size = 16
num_out = 3 # number of output result


# train data load
train_queue = tf.train.string_input_producer([train_csv_dir])
train_reader = tf.TextLineReader()
_, train_csv_value = train_reader.read(train_queue)
train_img_dir, train_label, train_gender = tf.decode_csv(train_csv_value, record_defaults=[[""], [-1], [-1]])
train_img_value = tf.read_file(train_img_dir)
train_img = tf.reshape(tf.cast(tf.image.decode_jpeg(train_img_value, channels=3), dtype=tf.float32), shape=[image_height, image_width, 3])
train_label = tf.reshape(tf.one_hot(train_label, depth=num_out, on_value=1.0, off_value=0.0), shape=[num_out])
train_gender = tf.reshape(train_gender, shape=[1])

# test data load
test_queue = tf.train.string_input_producer([test_csv_dir], shuffle=False)
test_reader = tf.TextLineReader()
_, test_csv_value = test_reader.read(test_queue)
test_img_dir, test_label, test_gender = tf.decode_csv(test_csv_value, record_defaults=[[""], [-1], [-1]])
test_img_value = tf.read_file(test_img_dir)
test_img = tf.reshape(tf.cast(tf.image.decode_jpeg(test_img_value, channels=3), dtype=tf.float32), shape=[image_height, image_width, 3])
test_label = tf.reshape(tf.one_hot(test_label, depth=num_out, on_value=1.0, off_value=0.0), shape=[num_out])
test_gender = tf.reshape(test_gender, shape=[1])

########################################################################################################################

class inceptionV3:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.keep_prob = tf.placeholder(dtype=tf.float32)  # drop-out %
            self.task = tf.placeholder(dtype=tf.bool)  # if true : training / if false : testing
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, 3])  # for image
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, num_out])  # for label

            y_ = tf.nn.relu(batch_norm(tf.layers.conv2d(self.x, 32, [3,3], (2,2), padding="valid"), 32, self.task))
            y_ = tf.nn.relu(batch_norm(tf.layers.conv2d(y_, 32, [3,3], padding='valid'), 32, self.task))
            y_ = tf.nn.relu(batch_norm(tf.layers.conv2d(y_, 64, [3,3], padding="SAME"), 64, self.task))
            y_ = tf.nn.max_pool(y_, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='valid')

            y_ = tf.nn.relu(batch_norm(tf.layers.conv2d(y_, 80, [1,1], padding='valid'), 80, self.task))
            y_ = tf.nn.relu(batch_norm(tf.layers.conv2d(y_, 192, [3,3], padding='valid'), 192, self.task))
            y_ = tf.nn.max_pool(y_, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='valid')

            # mixed 0, 1, 2: 35 x 35 x 256
            branch1x1 = tf.nn.relu(batch_norm(tf.layers.conv2d(y_, 64, [1,1], padding='SAME'), 64, self.task))

            branch5x5 = tf.nn.relu(batch_norm(tf.layers.conv2d(y_, 48, [1,1], padding='SAME'), 48, self.task))
            branch5x5 = tf.nn.relu(batch_norm(tf.layers.conv2d(branch5x5, 64, [5,5], padding='SAME'), 64, self.task))

            branch3x3dbl = tf.nn.relu(batch_norm(tf.layers.conv2d(y_, 64, [1,1], padding='SAME'), 64, self.task))
            branch3x3dbl = tf.nn.relu(batch_norm(tf.layers.conv2d(branch3x3dbl, 96, [3,3], padding='SAME'), 96, self.task))
            branch3x3dbl = tf.nn.relu(batch_norm(tf.layers.conv2d(branch3x3dbl, 96, [3,3], padding='SAME'), 96, self.task))

            branch_pool = tf.nn.avg_pool(y_, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
            branch_pool = tf.nn.relu(batch_norm(tf.layers.conv2d(branch_pool, 32, [1,1], padding='SAME'), 32, self.task))

            y_ = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name="1st_inception")

            # mixed 2: 35 x 35 x 256
            branch1x1 = tf.nn.relu(batch_norm(tf.layers.conv2d(y_, 64, [1,1], padding='SAME'), 64, self.task))

            branch5x5 = tf.nn.relu(batch_norm(tf.layers.conv2d(y_, 48, [1,1], padding='SAME'), 48, self.task))
            branch5x5 = tf.nn.relu(batch_norm(tf.layers.conv2d(branch5x5, 64, [5,5], padding='SAME'), 64, self.task))

            branch3x3dbl = tf.nn.relu(batch_norm(tf.layers.conv2d(y_, 64, [1,1], padding='SAME'), 64, self.task))
            branch3x3dbl = tf.nn.relu(batch_norm(tf.layers.conv2d(branch3x3dbl, 96, [3,3], padding='SAME'), 96, self.task))
            branch3x3dbl = tf.nn.relu(batch_norm(tf.layers.conv2d(branch3x3dbl, 96, [3,3], padding='SAME'), 96, self.task))

            branch_pool = tf.nn.avg_pool(y_, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
            branch_pool = tf.nn.relu(batch_norm(tf.layers.conv2d(branch_pool, 64, [1,1], padding='SAME'), 64, self.task))

            y_ = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name="2nd_inception")

            # mixed 3: 17 x 17 x 768
            branch3x3 = tf.nn.relu(batch_norm(tf.layers.conv2d(y_, 384, [3,3], strides=(2,2), padding='valid'), 384, self.task))

            branch3x3dbl = tf.nn.relu(batch_norm(tf.layers.conv2d(y_, 64, [1,1], padding='SAME'), 64, self.task))
            branch3x3dbl = tf.nn.relu(batch_norm(tf.layers.conv2d(branch3x3dbl, 96, [3,3], padding='SAME'), 96, self.task))
            branch3x3dbl = tf.nn.relu(batch_norm(tf.layers.conv2d(branch3x3dbl, 96, [3,3], strides=(2,2), padding='valid'), 96, self.task))

            branch_pool = tf.nn.max_pool(y_, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

            y_ = tf.concat([branch3x3, branch3x3dbl, branch_pool], axis=3, name="3nd_inception")




def weight(shape, name):
    #initial = tf.truncated_normal(shape, stddev=1e-1, dtype=tf.float32)
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    #return tf.Variable(initial, name=name)

def bias(shape, num, name):
    if num == 0.0: # conv-layer : initialie to 0.0
        return tf.Variable(tf.zeros(shape, dtype=tf.float32), name=name)
    else: # fully-connected layer : initialize to 1.0
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def conv(x, y, stride, padding="SAME"):
    return tf.nn.conv2d(x, y, strides=[1,stride,stride,1], padding=padding)

def batch_norm(batch_data, n_out, is_train):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(batch_data, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(batch_data, mean, var, beta, gamma, 1e-3)
    return normed

def batch_FC(inputs, is_train):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    batch_mean, batch_var = tf.nn.moments(inputs, [0])
    ema2 = tf.train.ExponentialMovingAverage(decay=0.99)

    def mean_var_with_update():
        ema2_apply_op = ema2.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema2_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(is_train, mean_var_with_update, lambda: (ema2.average(batch_mean), ema2.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, scale, 1e-3)
    return normed


