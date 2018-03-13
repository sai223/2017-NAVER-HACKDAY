import os
import tensorflow as tf
from functions import *


def list_tfrecord_file(file_list):
    tfrecord_list = []
    for i in range(len(file_list)):
        current_file_abs_path = os.path.abspath(file_list[i])
        if current_file_abs_path.endswith(".tfrecord"):
            tfrecord_list.append(current_file_abs_path)
            print("Found %s successfully!" % file_list[i])
        else:
            pass
    return tfrecord_list

def tfrecord_auto_traversal():
    current_folder_filename_list = os.listdir("./")
    if current_folder_filename_list != None:
        print("%s files were found under current folder. " % len(current_folder_filename_list))
        print("Please be noted that only files end with '*.tfrecord' will be load!")
        tfrecord_list = list_tfrecord_file(current_folder_filename_list)
        if len(tfrecord_list) != 0:
            for list_index in range(len(tfrecord_list)):
                print(tfrecord_list[list_index])
        else:
            print("Cannot find any tfrecord files, please check the path.")
    return tfrecord_list

def total_record_count():
    current_folder_filename_list = os.listdir("./")
    tfrecord_list = list_tfrecord_file(current_folder_filename_list)
    total_count = 0
    for path in tfrecord_list:
        total_count += sum(1 for _ in tf.python_io.tf_record_iterator(path))
    return total_count


image_height = 95
image_width = 95

num_out = 155695 # number of output result

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features = {
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/channels": tf.FixedLenFeature([], tf.int64),
        "image/class/label": tf.FixedLenFeature([], tf.int64),})

    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)

    image_shape = [95, 95, 3]
    image = tf.reshape(image_raw, image_shape)
    #image.set_shape(image_shape)
    image = tf.cast(image, tf.float32) * (1 / 255.0) - 0.5

    label = tf.cast(features["image/class/label"], tf.int32)
    label = tf.reshape(tf.one_hot(label, depth=num_out, on_value=1.0, off_value=0.0), shape=[num_out])
    filename = tf.cast(features["image/filename"], tf.string)
    return image, label, filename

num_epochs = 1
batch_size = 512
num_threads = 2
min_after_dequeue = 1000
capacity = min_after_dequeue + 3 * batch_size

filename_queue = tf.train.string_input_producer(tfrecord_auto_traversal(), num_epochs=num_epochs)

image, label, filename = read_and_decode(filename_queue)

example_batch, label_batch, filename_batch = tf.train.shuffle_batch([image, label, filename], batch_size=batch_size, 
    num_threads=num_threads, 
    capacity=capacity,
    min_after_dequeue=min_after_dequeue)

image_test = tf.placeholder(dtype=tf.float32, shape=[None, 95, 95, 3])
label_test = tf.placeholder(dtype=tf.int32, shape=[None])

img_shape = tf.shape(image_test)
lab_shape = tf.shape(label_test)

keep_prob = tf.placeholder(dtype=tf.float32) # drop-out %
x = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, 3]) # for image
y = tf.placeholder(dtype=tf.float32, shape=[None, num_out]) # for label


w_conv1_1 = weight([4, 4, 3, 128], 'w_conv1_1')
b_conv1_1 = bias([128], 0.0, 'b_conv1_1')

w_conv2_1 = weight([4, 4, 128, 256], 'w_conv2_1')
b_conv2_1 = bias([256], 0.0, 'b_conv2_1')

w_conv3_1 = weight([4, 4, 256, 512], 'w_conv3_1')
b_conv3_1 = bias([512], 0.0, 'b_conv3_1')

w_fc1 = weight([12 * 12 * 512, 2048], 'w_fc1')
b_fc1 = bias([2048], 1.0, 'b_fc1')
w_fc2 = weight([2048, 2048], 'w_fc2')
b_fc2 = bias([2048], 1.0, 'b_fc2')
w_vgg = weight([2048, num_out], 'w_vgg')
b_vgg = bias([num_out], 1.0, 'b_vgg')

y_label = tf.reshape(y, shape=[-1, num_out])

conv1_1 = tf.nn.tanh(tf.nn.bias_add(conv(x, w_conv1_1), b_conv1_1))
pool1 = tf.nn.max_pool(conv1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv2_1 = tf.nn.tanh(tf.nn.bias_add(conv(pool1, w_conv2_1), b_conv2_1))
pool2 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv3_1 = tf.nn.tanh(tf.nn.bias_add(conv(pool2, w_conv3_1), b_conv3_1))
pool3 = tf.nn.max_pool(conv3_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

flat = tf.reshape(pool3, [-1, 12 * 12 * 512])
fc1 = tf.nn.relu(tf.nn.dropout(tf.nn.bias_add(tf.matmul(flat, w_fc1), b_fc1), keep_prob=keep_prob))
fc2 = tf.nn.relu(tf.nn.dropout(tf.nn.bias_add(tf.matmul(fc1, w_fc2), b_fc2), keep_prob=keep_prob))
y_vgg = tf.nn.bias_add(tf.matmul(fc2, w_vgg), b_vgg)

label_value = tf.reshape(tf.cast(tf.argmax(y_label, 1), dtype=tf.int32), shape=[batch_size])
max_point = tf.reshape(tf.cast(tf.argmax(y_vgg, 1), dtype=tf.int32), shape=[batch_size])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_vgg))

# train
start_learning_rate = 0.001 # start_learning_rate
global_step = tf.Variable(0, trainable=False)
#tf.summary.scalar('loss', cross_entropy)
#learning_rate = tf.maximum(0.0001, tf.train.exponential_decay(start_learning_rate, global_step, 462, 0.8, staircase=True))
#train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
train = tf.train.AdamOptimizer(start_learning_rate).minimize(cross_entropy)

# accuracy
prediction = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_vgg, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
#tf.summary.scalar('accuracy', accuracy)

loss_log = []

# import pdb;pdb.set_trace()
with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=20)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    merge = tf.summary.merge_all()
    #temp_example, temp_label, temp_filename = sess.run([example_batch, label_batch, filename_batch])
    print("here1")
    #print(temp_example)
    #print(temp_label)
    #print(temp_filename)
    
    #image_shape, label_shape = sess.run([img_shape, lab_shape])
    #print(image_shape)
    #print(label_shape)
    
    now_step = 0
    
    for i in range(20000):
        batch_x, batch_y = sess.run([example_batch, label_batch])
        sess.run(train, feed_dict={keep_prob: 0.5, x: batch_x, y: batch_y})
        
        temp_loss = sess.run(cross_entropy, feed_dict={keep_prob: 1.0, x: batch_x, y: batch_y})
        #loss_log.append(temp_loss)
        acc = sess.run(accuracy, feed_dict={keep_prob: 1.0, x: batch_x, y: batch_y})
        if i % 50 == 0:
            saver.save(sess, './genre_ckpts/genre_Ckpt', global_step=now_step)
            now_step += 1
        print("i:", i)
        print("acc:", acc)
        print("loss:",temp_loss)
            
            
    
    #i_s, l_s, value = sess.run([img_shape, lab_shape, image_test], feed_dict={image_test: i_b, label_test: l_b})
    '''
    print("here2")
    print(i_s)
    print(l_s)
    print(value)
    print("here3")
    print("==========> total : ", total_record_count())
    '''
    
    coord.request_stop()
    coord.join(threads)





















