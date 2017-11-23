from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import argparse

import tensorflow as tf
from dir_traversal_tfrecord import tfrecord_auto_traversal
from dir_traversal_tfrecord import total_record_count

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
    # image_ = tf.decode_raw(features["image/encoded"], tf.uint8)
    #image_raw = tf.read_file(features["image/encoded"])
    #image = tf.reshape(tf.cast(tf.image.decode_jpeg(image_raw, channels=3), dtype=tf.float32), shape=[95, 95, 3])

    image_shape = [95, 95, 3]
    image_ = tf.reshape(image_raw, image_shape)
    #image_.set_shape([image_shape])
    image_ = tf.cast(image_, tf.float32) * (1. / 255.0) - 0.5

    label_ = tf.cast(features["image/class/label"], tf.int32)
    print("end randd!")
    return image_, label_


filename_queue = tf.train.string_input_producer(tfrecord_auto_traversal())

image, label = read_and_decode(filename_queue)
min_after_dequeue = 100000
capacity = 10000 #min_after_dequeue + 3 * 1

mini_batch_size = 512
min_queue_examples_train = 10000

example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=mini_batch_size, num_threads=8,
                                                  capacity=min_queue_examples_train + 8 * mini_batch_size,
                                                  min_after_dequeue=min_queue_examples_train)

image_test = tf.placeholder(dtype=tf.float32, shape=[None, 95, 95, 3])
label_test = tf.placeholder(dtype=tf.int32, shape=[None])

img_shape = tf.shape(image_test)
lab_shape = tf.shape(label_test)


# import pdb;pdb.set_trace()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
i_b, l_b = sess.run([example_batch, label_batch])
print("here1")
i_s, l_s, value = sess.run([img_shape, lab_shape, image_test], feed_dict={image_test: i_b, label_test: l_b})
#
print("here2")
print(i_s)
print(l_s)
print(value)
print("here3")
print("==========> total : ", total_record_count())

'''
config = tf.ConfigProto()
sess = tf.Session(config=config)
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

print('== Start training == ')
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    ep = 0
    step = 1
    while not coord.should_stop():
        import pdb;pdb.set_trace()
        eb, lb = sess.run([example_batch, label_batch])
except Exception:
    print("Exception!")
'''