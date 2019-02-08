import tensorflow as tf
import scipy
import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride,padding_='VALID'):
  return tf.nn.conv2d(x, W, strides=[1,stride, stride,1], padding=padding_)

x = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])
#keep_prob = tf.placeholder(tf.float32)

x_image = x

#first convolutional layer
W_conv1 = weight_variable([11, 11, 3, 96])
b_conv1 = bias_variable([96])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 4) + b_conv1)
print(h_conv1.shape)

# first max pool Layer 
max_pool_1 = tf.layers.max_pooling2d(inputs=h_conv1,pool_size=(3,3),strides=2,padding='VALID')
print(max_pool_1.shape)

#first Batch Normalization
norm_1 = tf.layers.batch_normalization(inputs = max_pool_1)

#second convolutional layer
W_conv2 = weight_variable([5, 5, 96, 256])
b_conv2 = bias_variable([256])

h_conv2 = tf.nn.relu(conv2d(norm_1, W_conv2, 1,padding_='SAME') + b_conv2)
print(h_conv2.shape)
# Second max pool Layer 
max_pool_2 = tf.layers.max_pooling2d(inputs=h_conv2,pool_size=(3,3),strides=2,padding='VALID')
print(max_pool_2.shape)

#second Batch Normalization
norm_2= tf.layers.batch_normalization(inputs = max_pool_2)

#third convolutional layer
W_conv3 = weight_variable([3, 3, 256, 384])
b_conv3 = bias_variable([384])
h_conv3 = tf.nn.relu(conv2d(norm_2, W_conv3, 1,padding_='SAME') + b_conv3)

print(h_conv3.shape)

#fourth convolutional layer
W_conv4 = weight_variable([3, 3, 384, 384])
b_conv4 = bias_variable([384])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1,padding_='SAME') + b_conv4)

print(h_conv4.shape)

#fourth convolutional layer
W_conv5 = weight_variable([3, 3, 384, 256])
b_conv5 = bias_variable([256])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1,padding_='SAME') + b_conv5)

print(h_conv5.shape)

# third max pool Layer 
max_pool_3 = tf.layers.max_pooling2d(inputs=h_conv5,pool_size=(3,3),strides=2)

print(max_pool_3.shape)

#First Dropout Layer
keep_prob_1 = tf.nn.dropout(x=max_pool_3,keep_prob=0.5)

tensor_shape = keep_prob_1.shape

#print('before flatten',tensor_shape)
#flatten_1
h_conv1_flat = tf.reshape(keep_prob_1, [-1, tensor_shape[1]*tensor_shape[2]*tensor_shape[3]])#find shape

#print(h_conv1_flat.shape)

#FCL 1
W_fc1 = weight_variable([9216, 4096])
b_fc1 = bias_variable([4096])
h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)

print(h_fc1.shape)
#keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=0.5)

#FCL 2
W_fc2 = weight_variable([4096, 4096])
b_fc2 = bias_variable([4096])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

print(h_fc2.shape)

#Output
W_output = weight_variable([4096, 1])
b_output = bias_variable([1])

y = tf.multiply(tf.atan(tf.matmul(h_fc2, W_output) + b_output), 2) #scale the atan output

