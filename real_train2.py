from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import argparse

import time as t
import tensorflow as tf
from dir_traversal_tfrecord import tfrecord_auto_traversal
from dir_traversal_tfrecord import total_record_count

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="3"  # graphic card number to use

image_height = 95
image_width = 95
train_batch_size = 512 # batch size
test_batch_size = 512
num_out = 212 # number of output result

keep_prob = tf.placeholder(dtype=tf.float32) # drop-out %
task = tf.placeholder(dtype=tf.bool) # if true : training / if false : testing
x = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, 3]) # for image
y = tf.placeholder(dtype=tf.float32, shape=[None, num_out]) # for label


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features = {
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/channels": tf.FixedLenFeature([], tf.int64),
        "image/class/label": tf.FixedLenFeature([], tf.int64)}) #tf.int64

    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)

    image_shape = [95, 95, 3]
    image_ = tf.reshape(image_raw, image_shape)
    image_ = tf.cast(image_, tf.float32) * (1. / 255.0) - 0.5

    label_ = tf.cast(features["image/class/label"], tf.int32)
    label_ = tf.reshape(tf.one_hot(label_, depth=num_out, on_value=1.0, off_value=0.0), shape=[num_out])
    return image_, label_

filename_queue = tf.train.string_input_producer(tfrecord_auto_traversal())

image, label = read_and_decode(filename_queue)


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


# batch_normalization function for conv-layer
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

# batch_normalization function for fully-connected layer
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


# convolution layers
w_conv1_1 = weight([3,3,3,64], 'w_conv1_1')
b_conv1_1 = bias([64], 0.0, 'b_conv1_1')
w_conv1_2 = weight([3,3,64,64], 'w_conv1_2')
b_conv1_2 = bias([64], 0.0, 'b_conv1_2')

w_conv1_3 = weight([1,1,64,64], 'w_conv1_3')
b_conv1_3 = bias([64], 0.0, 'b_conv1_3')

w_conv2_1 = weight([3,3,64,128], 'w_conv2_1')
b_conv2_1 = bias([128], 0.0, 'b_conv2_1')
w_conv2_2 = weight([3,3,128,128], 'w_conv2_2')
b_conv2_2 = bias([128], 0.0, 'b_conv2_2')
w_conv3_1 = weight([3,3,128,256], 'w_conv3_1')
b_conv3_1 = bias([256], 0.0 , 'b_conv3_1')
w_conv3_2 = weight([3,3,256,256], 'w_conv3_2')
b_conv3_2 = bias([256], 0.0, 'b_conv3_2')
w_conv3_3 = weight([3,3,256,256], 'w_conv3_3')
b_conv3_3 = bias([256], 0.0, 'b_conv3_3')
w_conv4_1 = weight([3,3,256,512], 'w_conv4_1')
b_conv4_1 = bias([512], 0.0, 'b_conv4_1')
w_conv4_2 = weight([3,3,512,512], 'w_conv4_2')
b_conv4_2 = bias([512], 0.0, 'b_conv4_2')
w_conv4_3 = weight([3, 3, 512, 512], 'w_conv4_3')
b_conv4_3 = bias([512], 0.0, 'b_conv4_3')
'''w_conv5_1 = weight([3,3,512,512], 'w_conv5_1')
b_conv5_1 = bias([512], 0.0, 'b_conv5_1')
w_conv5_2 = weight([3,3,512,512], 'w_conv5_2')
b_conv5_2 = bias([512], 0.0, 'b_conv5_2')
w_conv5_3 = weight([3,3,512,512], 'w_conv5_3')
b_conv5_3 = bias([512], 0.0, 'b_conv5_3')'''

# fully connected layers
w_fc1 = weight([6*6*512, 4096], 'w_fc1')
b_fc1 = bias([4096], 1.0, 'b_fc1')
w_fc2 = weight([4096, 4096], 'w_fc2')
b_fc2 = bias([4096], 1.0, 'b_fc2')
w_vgg = weight([4096, num_out], 'w_vgg')
b_vgg = bias([num_out], 1.0, 'b_vgg')


#x_image = tf.reshape(x, shape=[-1, 224, 224, 3])
y_label = tf.reshape(y, shape=[-1, num_out])

conv1_1 = tf.nn.relu(batch_norm((conv(x, w_conv1_1) + b_conv1_1), 64, task))
conv1_2 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(conv1_1, w_conv1_2), b_conv1_2), 64, task))
pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv2_1 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(pool1, w_conv2_1), b_conv2_1),128, task))
conv2_2 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(conv2_1, w_conv2_2), b_conv2_2), 128, task))
pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv3_1 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(pool2, w_conv3_1), b_conv3_1), 256, task))
conv3_2 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(conv3_1, w_conv3_2), b_conv3_2), 256, task))
conv3_3 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(conv3_2, w_conv3_3), b_conv3_3), 256, task))
pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv4_1 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(pool3, w_conv4_1), b_conv4_1), 512, task))
conv4_2 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(conv4_1, w_conv4_2), b_conv4_2), 512, task))
conv4_3 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(conv4_2, w_conv4_3), b_conv4_3), 512, task))
pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

'''conv5_1 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(pool4, w_conv5_1), b_conv5_1), 512, task))
conv5_2 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(conv5_1, w_conv5_2), b_conv5_2), 512, task))
conv5_3 = tf.nn.relu(batch_norm(tf.nn.bias_add(conv(conv5_2, w_conv5_3), b_conv5_3), 512, task))
pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')'''

flat = tf.reshape(pool4, [-1, 6 * 6 * 512])
fc1 = tf.nn.relu(batch_FC(tf.nn.dropout(tf.nn.bias_add(tf.matmul(flat, w_fc1), b_fc1), keep_prob=keep_prob), task))
fc2 = tf.nn.relu(batch_FC(tf.nn.dropout(tf.nn.bias_add(tf.matmul(fc1, w_fc2), b_fc2), keep_prob=keep_prob), task))
y_vgg = tf.nn.bias_add(tf.matmul(fc2, w_vgg), b_vgg)


label_value = tf.reshape(tf.cast(tf.argmax(y_label, 1), dtype=tf.int32), shape=[test_batch_size])
max_point = tf.reshape(tf.cast(tf.argmax(y_vgg, 1), dtype=tf.int32), shape=[test_batch_size])

# cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_vgg)) # better to use softmax func
#cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_label, logits=y_out))

# train
start_learning_rate = 0.0008 # start_learning_rate
global_step = tf.Variable(0, trainable=False)
tf.summary.scalar('loss', cross_entropy)
learning_rate = tf.maximum(0.0001, tf.train.exponential_decay(start_learning_rate, global_step, 231, 0.75, staircase=True))
train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

# accuracy
prediction = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_vgg, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

min_after_dequeue = 10000
capacity = 10000 #min_after_dequeue + 3 * 1

mini_batch_size = train_batch_size
min_queue_examples_train = 10000

example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=mini_batch_size, num_threads=8,
                                                  capacity=min_queue_examples_train + 8 * mini_batch_size,
                                                  min_after_dequeue=min_queue_examples_train, allow_smaller_final_batch=True)


with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=20)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, '/mnt/hdd3t/Data/hci1/hoon/2th/shape/total/2thCircleCkpts/CircleCkpt-50')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    merge = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./genre_summary/', sess.graph)
    now_epoch = 0

    for i in range(4621):
        batch_x, batch_y = sess.run([example_batch, label_batch])
        if i % 100 != 0:
            l, _ = sess.run([cross_entropy, train], feed_dict={keep_prob: 0.5, x: batch_x, y: batch_y, task: True})
            print("=== step : %s , loss = %s ==== " % (i, l))

        elif i % 100 == 0:
            _, acc, loss, mer, lr = sess.run([train, accuracy, cross_entropy, merge, learning_rate], feed_dict={keep_prob: 0.5, x: batch_x, y: batch_y, task: True})
            print("epoch  ", now_epoch)
            print("   LR = ", lr)
            print("   loss = ", loss)
            print("   accuracy = " + str(acc*100) + " %")
            currentTime = t.localtime()
            print(t.strftime("%Y-%m-%d %H:%M:%S", currentTime))
            train_writer.add_summary(mer, now_epoch)
            saver.save(sess, './genre_ckpts/genre_Ckpt', global_step=now_epoch)
            if i % 462 == 0:
                now_epoch += 1
            print("--------------------------------------------------------------------------------")
            '''
            t_acc = 0.0
            t_acc2 = 0.0
            tAcc2= 0.0
            for j in range(54):
                test_batch_x, test_batch_y, test_batch_d, test_batch_z = sess.run([b_test_image, b_test_label, b_test_dir, b_test_gender])
                t_acc = 0.0
                count_list = [0, 0]
                tAcc, result, labels = sess.run([accuracy, max_point, label_value], feed_dict={keep_prob: 1.0, x: test_batch_x, y: test_batch_y, z: test_batch_z, task: False})
                for g in range(len(result)):
                    if result[g] == 0:
                        count_list[0] = count_list[0] + 1
                    else:
                        count_list[1] = count_list[1] + 1

                t_acc = float(float(max(count_list))/float(len(result)))
                t_acc2 = t_acc2 + t_acc
                tAcc2 = tAcc2 + tAcc

                if now_epoch % 20 == 0:
                    # print(result2)
                    print(result)
                    print(labels)
                    print(str(j) + " Test Data Accuracy = %-4.2f // Match Percent = %-4.2f" % (t_acc * 100, tAcc * 100))

            print("Total Test Data Accuracy = %-4.2f // Original Accuracy = %-4.2f" % (t_acc2/54*100, tAcc2/54*100))
            now_epoch = now_epoch + 5
            print("========================================================================================")
            print("========================================================================================")
            '''
    coord.request_stop()
    coord.join(threads)
