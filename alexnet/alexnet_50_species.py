# coding: utf-8
from __future__ import print_function
import scipy.io
import tensorflow as tf
from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib
import PIL
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt
import shutil

net_data = load("bvlc_alexnet.npy").item()

DROPOUT = 0.5
LEARNING_RATE = 0.01
VALIDATION_SIZE = 0
TRAINING_ITERATIONS = 100000
WEIGHT_DECAY = 0.0005
NB_EPOCH = 200
batch_size = 50
num_classes = 50
label_id = 0

train_dir = list() # list directory of train images
train_labels = list() # list label of train images
train_dir_tmp = list() # it is used for shuffling train data
train_labels_tmp = list()
svm_dir = list() # list directory of svm images
svm_labels = list() # list label of svm images
test_dir = list() # list directory of test images
test_labels = list() # list label of test images
species_dict = dict() # dictionary for species id. It map real species id to new neat id ([0, 1, 2, ..., 49])

opt = sys.argv[1] #option
organ = sys.argv[2] # leaf, flower, entire or branch

if (opt == '--organ'):
    if (organ not in ['leaf', 'flower', 'entire', 'branch']):
        sys.exit('Plant argument must be leaf, flower, entire or branch')
else:
    sys.exit('option ' + opt + ' does not exist, please choose: --organ')

data_folder = '../plant_data/' + organ
file_id = '50_' + organ + '_pretrained'
flag_train = False

for sub_dir_flower in os.listdir(data_folder + '/Training/'):
    for img in os.listdir(data_folder + '/Training/' + sub_dir_flower):
        try:
            pic = Image.open(os.path.join(data_folder + '/Training/' + sub_dir_flower, img))
        except IOError:
            continue
        if (pic.format != 'JPEG'):
            continue
            # rgb_pic = pic.convert('RGB')
            # rgb_pic.save(os.path.join(data_folder + '/Training/' + sub_dir_flower, img))

        train_dir_tmp.append(os.path.join(data_folder + '/Training/' + sub_dir_flower, img))
        train_labels_tmp.append(label_id)

    species_dict[sub_dir_flower] = label_id
    label_id = label_id + 1
print(species_dict)
index_shuf = range(len(train_dir_tmp))
shuffle(index_shuf)
for i in index_shuf:
    train_dir.append(train_dir_tmp[i])
    train_labels.append(train_labels_tmp[i])


for sub_dir_flower in os.listdir(data_folder + '/SvmInput/'):
    for img in os.listdir(data_folder + '/SvmInput/' + sub_dir_flower):
        pic = Image.open(os.path.join(data_folder + '/SvmInput/' + sub_dir_flower, img))
        if (pic.format != 'JPEG'):
            print((os.path.join(data_folder + '/SvmInput/' + sub_dir_flower, img)))
            # rgb_pic = pic.convert('RGB')
            # rgb_pic.save(os.path.join(data_folder + '/SvmInput/' + sub_dir_flower, img))
        else:
            svm_dir.append(os.path.join(data_folder + '/SvmInput/' + sub_dir_flower, img))
            svm_labels.append(species_dict[sub_dir_flower])

for sub_dir_flower in os.listdir(data_folder + '/Testing/'):
    for img in os.listdir(data_folder + '/Testing/' + sub_dir_flower):
        pic = Image.open(os.path.join(data_folder + '/Testing/' + sub_dir_flower, img))
        if (pic.format != 'JPEG'):
            print((os.path.join(data_folder + '/Testing/' + sub_dir_flower, img)))
            # rgb_pic = pic.convert('RGB')
            # rgb_pic.save(os.path.join(data_folder + '/Testing/' + sub_dir_flower, img))
        else:
            test_dir.append(os.path.join(data_folder + '/Testing/' + sub_dir_flower, img))
            test_labels.append(species_dict[sub_dir_flower])
			
# Un-comment this block if you want to use an available svm and test directory
'''
svm_dir = list() # list directory of svm images
svm_labels = list()	# list label of svm images
test_dir = list() # list directory of test images
test_labels = list() # list label of test images

f = open(organ + '_test', 'r')
for line in f:
    test_dir.append(line.split(';')[0])
    test_labels.append(species_dict[line.strip().split(';')[1]])
f.close()

f = open(organ + '_svm', 'r')
for line in f:
    svm_dir.append(line.split(';')[0])
    svm_labels.append(species_dict[line.strip().split(';')[1]])
f.close()
'''

print(len(train_dir))
print(len(test_dir))
print(len(svm_dir))

def print_activations(t):
    '''
	print a tensor shape
	:param t: is a tensor
	'''
    print(t.op.name, ' ', t.get_shape().as_list())

def dense_to_one_hot(labels_dense, num_classes):
    '''
	make the one-hot matrix for label list
	input: a list of label id. For example [1, 2, 3]
	output: a one-hot maxtrix. [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
	Note: you can do it in the tensorflow model, but I want to do it here because 
	I use it for feeding the model. Therefor, my model can be sped up so much.
	
	:param t: is a tensor
	'''
	
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def read_images_from_disk(input_queue):
    '''
	This function is used for reading images to tensors vector in tensorflow.
	Input: a (directory, label) queue of some images
	Output: a tensor vector for each image; a label list for each image
	
	:param input_queue: (directory, label) queue
	'''
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)  #read image with jpge extension
    return example, label

# weight initialization
def weight_variable(shape, name):
    '''
	init weight variable for CNN model
	Input: shape and name of our variable
	Ouput:
	
	:param shape: the shape of the expect variable
	:param shape: the name of the expect variable
	'''
    initial = tf.truncated_normal(shape, stddev=0.01, name=name)
    return tf.Variable(initial)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)

# convolution
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

def conv2d(x, W, stride_h, stride_w, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride_h, stride_w, 1], padding=padding)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

num_examples = len(train_dir)
train_accuracies = list()
train_costs = list()
validation_accuracies = list()
x_range = list()

# CNN modeling
graph = tf.Graph()
with graph.as_default():
    y_test = np.asarray(test_labels)
    y_valid = np.asarray(svm_labels)
    y_train = np.asarray(train_labels)

    y_test = dense_to_one_hot(y_test, num_classes)
    y_valid = dense_to_one_hot(y_valid, num_classes)
    y_train = dense_to_one_hot(y_train, num_classes)

    x_test = test_dir
    x_valid = svm_dir
    x_train = train_dir

    input_queue_test = tf.train.slice_input_producer([x_test, y_test],
                                                     num_epochs=None,
                                                     shuffle=False)

    x_test, y_test = read_images_from_disk(input_queue_test)
    x_test = tf.image.resize_images(x_test, [227, 227], method=1)
    # x_test = tf.image.per_image_whitening(x_test)

    x_test, y_test = tf.train.batch([x_test, y_test], batch_size=len(test_dir), allow_smaller_final_batch=True)

    input_queue_valid = tf.train.slice_input_producer([x_valid, y_valid],
                                                      num_epochs=None,
                                                      shuffle=False)

    x_valid, y_valid = read_images_from_disk(input_queue_valid)
    x_valid = tf.image.resize_images(x_valid, [227, 227], method=1)
    x_valid, y_valid = tf.train.batch([x_valid, y_valid], batch_size=batch_size, allow_smaller_final_batch=True)

    input_queue_train = tf.train.slice_input_producer([x_train, y_train],
                                                      num_epochs=None,
                                                      shuffle=True)

    x_train, y_train = read_images_from_disk(input_queue_train)
    x_train = tf.image.resize_images(x_train, [227, 227], method=1)
    # x_train_rot = tf.image.rot90(x_train, k=1)
    # x_train = tf.image.per_image_whitening(x_train)

    x_train, y_train = tf.train.shuffle_batch([x_train, y_train], batch_size=batch_size, num_threads=4,
                                                          capacity=5000, min_after_dequeue=1000,
                                                          allow_smaller_final_batch=True)

    x = tf.placeholder('float', shape=[None, 227, 227, 3])
    y_ = tf.placeholder('float', shape=[None, num_classes])
    x_testdata = tf.placeholder('float', shape=[None, 227, 227, 3])
    y_testdata = tf.placeholder('float', shape=[None, num_classes])

    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    fc6W = tf.Variable(net_data["fc6"][0])
    fc6b = tf.Variable(net_data["fc6"][1])
    fc7W = tf.Variable(net_data["fc7"][0])
    fc7b = tf.Variable(net_data["fc7"][1])
    fc8W = weight_variable([4096, num_classes], 'W_fc8')
    fc8b = bias_variable([num_classes], 'b_fc8')
    keep_prob = tf.placeholder('float')

    def model(x):
        # conv1
        # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11
        k_w = 11
        c_o = 96
        s_h = 4
        s_w = 4
        conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)

        # lrn1
        # lrn(2, 2e-05, 0.75, name='norm1')
        radius = 5
        alpha = 0.0001
        beta = 0.75
        bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

        # maxpool1
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        # conv2
        # conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5
        k_w = 5
        c_o = 256
        s_h = 1
        s_w = 1
        group = 2
        conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)

        # lrn2
        # lrn(2, 2e-05, 0.75, name='norm2')
        radius = 5
        alpha = 0.0001
        beta = 0.75
        bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

        # maxpool2
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        # conv3
        # conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3
        k_w = 3
        c_o = 384
        s_h = 1
        s_w = 1
        group = 1

        conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)

        # conv4
        # conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3
        k_w = 3
        c_o = 384
        s_h = 1
        s_w = 1
        group = 2
        conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)

        # conv5
        # conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3
        k_w = 3
        c_o = 256
        s_h = 1
        s_w = 1
        group = 2
        conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)

        # maxpool5
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        # fc6
        # fc(4096, name='fc6')
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
        fc6_drop = tf.nn.dropout(fc6, keep_prob)

        # fc7
        # fc(4096, name='fc7')
        fc7 = tf.nn.relu_layer(fc6_drop, fc7W, fc7b)
        fc7_drop = tf.nn.dropout(fc7, keep_prob)
        # fc8
        # fc(1000, relu=False, name='fc8')
        fc8 = (tf.nn.xw_plus_b(fc7_drop, fc8W, fc8b))

        # prob
        # softmax(name='prob'))
        # prob = tf.nn.softmax(fc8)
        return fc8


    logits = model(x)
    # Choose softmax or sigmoid cross entropy
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))

    regularizers = tf.nn.l2_loss(conv1W) + tf.nn.l2_loss(conv1b) + \
                   tf.nn.l2_loss(conv2W) + tf.nn.l2_loss(conv2b) + \
                   tf.nn.l2_loss(conv3W) + tf.nn.l2_loss(conv3b) + \
                   tf.nn.l2_loss(conv4W) + tf.nn.l2_loss(conv4b) + \
                   tf.nn.l2_loss(conv5W) + tf.nn.l2_loss(conv5b) + \
                   tf.nn.l2_loss(fc6W) + tf.nn.l2_loss(fc6b) + \
                   tf.nn.l2_loss(fc7W) + tf.nn.l2_loss(fc7b) + \
                   tf.nn.l2_loss(fc8W) + tf.nn.l2_loss(fc8b)

    loss = tf.reduce_mean(cross_entropy + WEIGHT_DECAY * regularizers)
	
    # optimisation function
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, 1000, 0.65, staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    saver = tf.train.Saver()

    with tf.device('/cpu:0'):
        logits_test = model(x_testdata)

        prediction_vector_score = tf.nn.softmax(logits_test)
        prediction_test = tf.argmax(prediction_vector_score, 1)
        accuracy_test = tf.reduce_mean(tf.cast(tf.equal(prediction_test, tf.argmax(y_testdata, 1)), 'float'))
        top_5_correct_prediction = tf.nn.in_top_k(tf.nn.softmax(logits_test), tf.argmax(y_testdata, 1), k=5)
        top_5_accuracy = tf.reduce_mean(tf.cast(top_5_correct_prediction, 'float'))

iter_per_epoch = len(train_dir) / batch_size + 1
# Make a session for training CNN model
print('Training ...')

with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    saver.restore(sess, '../ckpt_alexnet/alex_model_' + file_id + '.ckpt')

    if (flag_train):
        for i in range(TRAINING_ITERATIONS):
            xtrain, ytrain = sess.run([x_train, y_train])
            _, train_accuracy, cost =  sess.run([train_step, accuracy, loss], feed_dict={x: xtrain, y_: ytrain, keep_prob: 0.5})

            print('training_accuracy => %.3f, cost value => %.5f for step %d, learning_rate => %.5f' % (
                train_accuracy, cost, i, learning_rate.eval()))

            if i % iter_per_epoch == 0:
                train_accuracies.append(train_accuracy)
                train_costs.append(cost)

            if i // iter_per_epoch > NB_EPOCH:
                break

        plt.plot(train_accuracies)
        axes = plt.gca()
        axes.set_ylim([0, 1.2])
        plt.title(file_id + ' batch train accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('../chart/accuracy_' + file_id + '.png')
        plt.close()

        plt.plot(train_costs)
        plt.title(file_id + ' batch train loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('../chart/loss_' + file_id + '.png')
        plt.close()
        saver.save(sess, '../ckpt_alexnet/alex_model_' + file_id + '.ckpt')

    xtest, ytest = sess.run([x_test, y_test])
    test_accuracy, test_prediction, test_vector_score, test_vector_logits, test_top_5_accuracy = sess.run(
        [accuracy_test, prediction_test, prediction_vector_score, logits_test, top_5_accuracy],
        feed_dict={x_testdata: xtest,
                   y_testdata: ytest,
                   keep_prob: 1.0})
    del xtest

    fs = open('../result/test_result_' + file_id + '.txt', 'w')
    fl = open('../result/test_result_' + file_id + '_logits.txt', 'w')

    for it in range(len(test_dir)):
        for itt in range(len(test_vector_score[it])):
            fs.write(test_dir[it] + ' ' + str(test_labels[it]) + ' ' + str(itt + 1) + ' ' + str(
                test_vector_score[it][itt]))
            fs.write('\n')
            fl.write(test_dir[it] + ' ' + str(test_labels[it]) + ' ' + str(itt + 1) + ' ' + str(
                test_vector_logits[it][itt]))
            fl.write('\n')
    fs.close()
    fl.close()

    fs = open('../result/svm_result_' + file_id + '.txt', 'w')
    fl = open('../result/svm_result_' + file_id + '_logits.txt', 'w')
    for it in range(len(svm_dir) // batch_size):
        xvalid, yvalid = sess.run([x_valid, y_valid])

        valid_accuracy, valid_prediction, valid_vector_score, valid_vector_logits = sess.run(
            [accuracy_test, prediction_test, prediction_vector_score, logits_test],
            feed_dict={x_testdata: xvalid,
                       y_testdata: yvalid,
                       keep_prob: 1.0})

        for k in range(batch_size):
            for itt in range(len(valid_vector_score[k])):
                fs.write(svm_dir[it * batch_size + k] + ' ' + str(svm_labels[it * batch_size + k]) + ' ' + \
                         str(itt + 1) + ' ' + str(valid_vector_score[k][itt]))
                fs.write('\n')

                fl.write(svm_dir[it * batch_size + k] + ' ' + str(svm_labels[it * batch_size + k]) + ' ' + \
                         str(itt + 1) + ' ' + str(valid_vector_logits[k][itt]))
                fl.write('\n')
    fs.close()
    fl.close()

    print('test_accuracy => %.3f' % test_accuracy)
    print('top 5 test_accuracy => %.3f' % test_top_5_accuracy)

    coord.request_stop()
    coord.join(threads)

sess.close()