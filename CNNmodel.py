
import os, datetime
import numpy as np
import tensorflow as tf
import tflearn
from tensorflow.contrib.layers.python.layers import batch_norm
from PIL import Image
import tflearn
import random
import scipy
from time import time
import sys

image_size = 512


def create_conv_layer(name, input_tensor, in_channels, out_channels, is_training = True, activation='relu', kx = 3, ky = 3, stride_x = 2, stride_y = 2, batchnorm=False, padding='VALID', add=None, deconv = False):
	if deconv == False:
		input_tensor = tf.pad(input_tensor, [[0, 0], [kx/2, kx/2], [kx/2, kx/2], [0, 0]], mode="CONSTANT")


	weights = tf.get_variable(name+'weights', shape=[kx, ky, in_channels, out_channels],
			initializer=tf.truncated_normal_initializer(stddev=np.sqrt(0.02 / kx / ky / in_channels)),
			dtype=tf.float32
	)
	biases = tf.get_variable(name+'biases', shape=[out_channels], initializer=tf.constant_initializer(0.0), dtype=tf.float32)



	if deconv == False:
		t = tf.nn.conv2d(input_tensor, weights, [1, stride_x, stride_y, 1], padding=padding)
		s = tf.nn.bias_add(t, biases)

	else:
		batch = tf.shape(input_tensor)[0]
		size = tf.shape(input_tensor)[1]


		print(input_tensor)
		print(tf.transpose(weights,perm=[0,1,3,2]))



		t = tf.nn.conv2d_transpose(input_tensor, tf.transpose(weights,perm=[0,1,3,2]),[batch, size * stride_x, size * stride_y, out_channels], [1, stride_x, stride_y, 1],
				padding='SAME', data_format = "NHWC")
		
		# t = tf.nn.conv2d_transpose(input_tensor, tf.transpose(weights,perm=[0,1,3,2]),tf.tensor([batch, size * stride_x, size * stride_y, out_channels]), [1, stride_x, stride_y, 1],
		# 		padding='SAME', data_format = "NHWC")
		

		s = tf.nn.bias_add(t, biases)

	if add is not None: # res
		s = s + add 

	if batchnorm:
		n = batch_norm(s, decay = 0.99, center=True, scale=True, updates_collections=None, is_training=is_training)
	else:
		n = s 

	if activation == 'relu':
			return tf.nn.relu(n), weights, biases
	elif activation == 'sigmoid':
			return tf.nn.sigmoid(n), weights, biases
	elif activation == 'tanh':
			return tf.nn.tanh(n), weights, biases
	elif activation == 'linear':
			return n, weights, biases

def forward_conv_layer(name, input_tensor, in_channels, out_channels, is_training = True, activation='relu', kx = 3, ky = 3, stride_x = 2, stride_y = 2, batchnorm=False, padding='VALID', add=None, w = None, b = None, deconv = False):
	if deconv == False:
		input_tensor = tf.pad(input_tensor, [[0, 0], [kx/2, kx/2], [kx/2, kx/2], [0, 0]], mode="CONSTANT")

	weights  = w
	biases  = b

	if deconv == False:
		t = tf.nn.conv2d(input_tensor, weights, [1, stride_x, stride_y, 1], padding=padding)
		s = tf.nn.bias_add(t, biases)

	else:
		batch = tf.shape(input_tensor)[0]
		size = tf.shape(input_tensor)[1]

		t = tf.nn.conv2d_transpose(input_tensor, tf.transpose(weights,perm=[0,1,3,2]),[batch, size * stride_x, size * stride_y, out_channels], [1, stride_x, stride_y, 1],
                    padding='SAME')
		
		s = tf.nn.bias_add(t, biases)




	if add is not None: # res
		s = s + add 

	if batchnorm:
		n = batch_norm(s, decay = 0.99, center=True, scale=True, updates_collections=None, is_training=is_training)
	else:
		n = s 

	if activation == 'relu':
			return tf.nn.relu(n)
	elif activation == 'sigmoid':
			return tf.nn.sigmoid(n)
	elif activation == 'tanh':
			return tf.nn.tanh(n)
	elif activation == 'linear':
			return n






def build_unet512_10(inputs, inputdim = 3, prefix = "none_",last_activation = 'linear', deconv = False):
	vs = {}

	conv1,vs['w1'],vs['b1'] = create_conv_layer(prefix+'unet512_1', inputs, inputdim, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5)
	conv2,vs['w2'],vs['b2'] = create_conv_layer(prefix+'unet512_2', conv1, 32, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5)
	conv3,vs['w3'],vs['b3'] = create_conv_layer(prefix+'unet512_3', conv2, 64, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128
	conv4,vs['w4'],vs['b4'] = create_conv_layer(prefix+'unet512_4', conv3, 64, 128, stride_x = 2, stride_y = 2, kx = 5, ky = 5) # 64 * 64

	#conv4_up = tf.image.resize_images(conv4, [128, 128], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	if deconv == False:
		conv4_up = tf.image.resize_nearest_neighbor(conv4, [128, 128])
		#conv4_up = imageUpSampling2x(conv4)
		conv5,vs['w5'],vs['b5'] = create_conv_layer(prefix+'unet512_5', conv4_up, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128
	else:
		conv5,vs['w5'],vs['b5'] = create_conv_layer(prefix+'unet512_5', conv4, 128, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, deconv = True) # 128 * 128
	


	conv5_concat = tf.concat([conv5 , conv3],3)


	conv6,vs['w6'],vs['b6'] = create_conv_layer(prefix+'unet512_6', conv5_concat, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128
	conv7,vs['w7'],vs['b7'] = create_conv_layer(prefix+'unet512_7', conv6, 64, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128

	if deconv == False:
		conv7_up = tf.image.resize_images(conv7, [256, 256], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		#conv7_up = imageUpSampling2x(conv7)
		conv8,vs['w8'],vs['b8'] = create_conv_layer(prefix+'unet512_8', conv7_up, 32, 16, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 512 * 512
		conv8_up = tf.image.resize_images(conv8, [512, 512], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		#conv8_up = imageUpSampling2x(conv8)
		conv9,vs['w9'],vs['b9'] = create_conv_layer(prefix+'unet512_9', conv8_up, 16, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 512 * 512
	else:
		conv8,vs['w8'],vs['b8'] = create_conv_layer(prefix+'unet512_8', conv7, 32, 16, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True) # 512 * 512
		conv9,vs['w9'],vs['b9'] = create_conv_layer(prefix+'unet512_9', conv8, 16, 8, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True) # 512 * 512
	

	conv10,vs['w10'],vs['b10'] = create_conv_layer(prefix+'unet512_10', conv9, 8, 2, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation) # 512 * 512
	
	return conv10, vs

def run_unet512_10(inputs, weights, inputdim = 3, prefix = "none_",last_activation = 'linear',deconv = False):
	#print(weights['w1'], weights['b1'])
	conv1= forward_conv_layer(prefix+'unet512_1', inputs, inputdim, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w1'], b=weights['b1'])
	#print(conv1)
	conv2= forward_conv_layer(prefix+'unet512_2', conv1, 32, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w2'], b=weights['b2'])
	conv3= forward_conv_layer(prefix+'unet512_3', conv2, 64, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w3'], b=weights['b3']) # 128 * 128
	conv4= forward_conv_layer(prefix+'unet512_4', conv3, 64, 128, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w4'], b=weights['b4']) # 64 * 64

	#conv4_up = tf.image.resize_images(conv4, [128, 128], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	
	if deconv == False:
		conv4_up = tf.image.resize_nearest_neighbor(conv4, [128, 128])
		#conv4_up = imageUpSampling2x(conv4)

		conv5= forward_conv_layer(prefix+'unet512_5', conv4_up, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w5'], b=weights['b5']) # 128 * 128
	else:
		conv5= forward_conv_layer(prefix+'unet512_5', conv4, 128, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w5'], b=weights['b5'], deconv = True) # 128 * 128
	

	conv5_concat = tf.concat([conv5 , conv3],3)


	conv6= forward_conv_layer(prefix+'unet512_6', conv5_concat, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w6'], b=weights['b6']) # 128 * 128
	conv7= forward_conv_layer(prefix+'unet512_7', conv6, 64, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w7'], b=weights['b7']) # 128 * 128

	if deconv == False:
		conv7_up = tf.image.resize_images(conv7, [256, 256], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		#conv7_up = imageUpSampling2x(conv7)

		conv8= forward_conv_layer(prefix+'unet512_8', conv7_up, 32, 16, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w8'], b=weights['b8']) # 512 * 512
		conv8_up = tf.image.resize_images(conv8, [512, 512], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		#conv8_up = imageUpSampling2x(conv8)

		conv9= forward_conv_layer(prefix+'unet512_9', conv8_up, 16, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w9'], b=weights['b9']) # 512 * 512
	
	else:
		conv8= forward_conv_layer(prefix+'unet512_8', conv7, 32, 16, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w8'], b=weights['b8'], deconv=True) # 512 * 512
		conv9= forward_conv_layer(prefix+'unet512_9', conv8, 16, 8, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w9'], b=weights['b9'], deconv=True) # 512 * 512
	

	conv10= forward_conv_layer(prefix+'unet512_10', conv9, 8, 2, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation, w=weights['w10'], b=weights['b10']) # 512 * 512
	
	return conv10






def build_unet512_12(inputs, inputdim = 3, prefix = "none_",last_activation = 'linear', deconv = False):
	vs = {}

	conv1,vs['w1'],vs['b1'] = create_conv_layer(prefix+'unet512_1', inputs, inputdim, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5)
	conv2,vs['w2'],vs['b2'] = create_conv_layer(prefix+'unet512_2', conv1, 32, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5)
	conv3,vs['w3'],vs['b3'] = create_conv_layer(prefix+'unet512_3', conv2, 64, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128
	conv4,vs['w4'],vs['b4'] = create_conv_layer(prefix+'unet512_4', conv3, 64, 128, stride_x = 2, stride_y = 2, kx = 5, ky = 5) # 64 * 64

	#conv4_up = tf.image.resize_images(conv4, [128, 128], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	if deconv == False:
		conv4_up = tf.image.resize_nearest_neighbor(conv4, [128, 128])
		#conv4_up = imageUpSampling2x(conv4)
		conv5,vs['w5'],vs['b5'] = create_conv_layer(prefix+'unet512_5', conv4_up, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128
	else:
		conv5,vs['w5'],vs['b5'] = create_conv_layer(prefix+'unet512_5', conv4, 128, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, deconv = True) # 128 * 128
	


	conv5_concat = tf.concat([conv5 , conv3],3)


	conv6,vs['w6'],vs['b6'] = create_conv_layer(prefix+'unet512_6', conv5_concat, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128
	conv7,vs['w7'],vs['b7'] = create_conv_layer(prefix+'unet512_7', conv6, 64, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128

	if deconv == False:
		conv7_up = tf.image.resize_images(conv7, [256, 256], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		#conv7_up = imageUpSampling2x(conv7)
		conv8,vs['w8'],vs['b8'] = create_conv_layer(prefix+'unet512_8', conv7_up, 32, 16, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 512 * 512
		conv8_up = tf.image.resize_images(conv8, [512, 512], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		#conv8_up = imageUpSampling2x(conv8)
		conv9,vs['w9'],vs['b9'] = create_conv_layer(prefix+'unet512_9', conv8_up, 16, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 512 * 512
	else:
		conv8,vs['w8'],vs['b8'] = create_conv_layer(prefix+'unet512_8', conv7, 32, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True) # 512 * 512
		conv9,vs['w9'],vs['b9'] = create_conv_layer(prefix+'unet512_9', conv8, 32, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True) # 512 * 512
	

	conv10,vs['w10'],vs['b10'] = create_conv_layer(prefix+'unet512_10', conv9, 32, 16, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation) # 512 * 512
	conv11,vs['w11'],vs['b11'] = create_conv_layer(prefix+'unet512_11', conv10, 16, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation) # 512 * 512
	conv12,vs['w12'],vs['b12'] = create_conv_layer(prefix+'unet512_12', conv11, 8, 2, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation) # 512 * 512
	
	return conv12, vs

def run_unet512_12(inputs, weights, inputdim = 3, prefix = "none_",last_activation = 'linear',deconv = False):
	#print(weights['w1'], weights['b1'])
	conv1= forward_conv_layer(prefix+'unet512_1', inputs, inputdim, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w1'], b=weights['b1'])
	#print(conv1)
	conv2= forward_conv_layer(prefix+'unet512_2', conv1, 32, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w2'], b=weights['b2'])
	conv3= forward_conv_layer(prefix+'unet512_3', conv2, 64, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w3'], b=weights['b3']) # 128 * 128
	conv4= forward_conv_layer(prefix+'unet512_4', conv3, 64, 128, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w4'], b=weights['b4']) # 64 * 64

	#conv4_up = tf.image.resize_images(conv4, [128, 128], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	
	if deconv == False:
		conv4_up = tf.image.resize_nearest_neighbor(conv4, [128, 128])
		#conv4_up = imageUpSampling2x(conv4)

		conv5= forward_conv_layer(prefix+'unet512_5', conv4_up, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w5'], b=weights['b5']) # 128 * 128
	else:
		conv5= forward_conv_layer(prefix+'unet512_5', conv4, 128, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w5'], b=weights['b5'], deconv = True) # 128 * 128
	

	conv5_concat = tf.concat([conv5 , conv3],3)


	conv6= forward_conv_layer(prefix+'unet512_6', conv5_concat, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w6'], b=weights['b6']) # 128 * 128
	conv7= forward_conv_layer(prefix+'unet512_7', conv6, 64, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w7'], b=weights['b7']) # 128 * 128

	if deconv == False:
		conv7_up = tf.image.resize_images(conv7, [256, 256], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		#conv7_up = imageUpSampling2x(conv7)

		conv8= forward_conv_layer(prefix+'unet512_8', conv7_up, 32, 16, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w8'], b=weights['b8']) # 512 * 512
		conv8_up = tf.image.resize_images(conv8, [512, 512], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		#conv8_up = imageUpSampling2x(conv8)

		conv9= forward_conv_layer(prefix+'unet512_9', conv8_up, 16, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w9'], b=weights['b9']) # 512 * 512
	
	else:
		conv8= forward_conv_layer(prefix+'unet512_8', conv7, 32, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w8'], b=weights['b8'], deconv=True) # 512 * 512
		conv9= forward_conv_layer(prefix+'unet512_9', conv8, 32, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w9'], b=weights['b9'], deconv=True) # 512 * 512
	

	conv10= forward_conv_layer(prefix+'unet512_10', conv9, 32, 16, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation, w=weights['w10'], b=weights['b10']) # 512 * 512
	conv11= forward_conv_layer(prefix+'unet512_11', conv10, 16, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation, w=weights['w11'], b=weights['b11']) # 512 * 512
	conv12= forward_conv_layer(prefix+'unet512_12', conv11, 8, 2, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation, w=weights['w12'], b=weights['b12']) # 512 * 512
	
	return conv12


def build_unet512_12_V1(inputs, inputdim = 3, prefix = "none_",last_activation = 'linear', deconv = False):
	vs = {}
	conv0,vs['w0'],vs['b0'] = create_conv_layer(prefix+'unet512_0', inputs, inputdim, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 256 * 256
	conv1,vs['w1'],vs['b1'] = create_conv_layer(prefix+'unet512_1', conv0, 32, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 256 * 256
	conv2,vs['w2'],vs['b2'] = create_conv_layer(prefix+'unet512_2', conv1, 32, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5) # 128 * 128 
	conv3,vs['w3'],vs['b3'] = create_conv_layer(prefix+'unet512_3', conv2, 64, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5) # 128 * 128
	conv4,vs['w4'],vs['b4'] = create_conv_layer(prefix+'unet512_4', conv3, 64, 128, stride_x = 2, stride_y = 2, kx = 5, ky = 5) # 64 * 64

	#conv4_up = tf.image.resize_images(conv4, [128, 128], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	if deconv == False:
		conv4_up = tf.image.resize_nearest_neighbor(conv4, [128, 128])
		#conv4_up = imageUpSampling2x(conv4)
		conv5,vs['w5'],vs['b5'] = create_conv_layer(prefix+'unet512_5', conv4_up, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128
	else:
		conv5,vs['w5'],vs['b5'] = create_conv_layer(prefix+'unet512_5', conv4, 128, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, deconv = True) # 128 * 128
	


	conv5_concat = tf.concat([conv5 , conv3],3)


	conv6,vs['w6'],vs['b6'] = create_conv_layer(prefix+'unet512_6', conv5_concat, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128
	conv7,vs['w7'],vs['b7'] = create_conv_layer(prefix+'unet512_7', conv6, 64, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128

	if deconv == False:
		conv7_up = tf.image.resize_images(conv7, [256, 256], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		#conv7_up = imageUpSampling2x(conv7)
		conv8,vs['w8'],vs['b8'] = create_conv_layer(prefix+'unet512_8', conv7_up, 32, 16, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 512 * 512
		conv8_up = tf.image.resize_images(conv8, [512, 512], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		#conv8_up = imageUpSampling2x(conv8)
		conv9,vs['w9'],vs['b9'] = create_conv_layer(prefix+'unet512_9', conv8_up, 16, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 512 * 512
	else:
		conv8,vs['w8'],vs['b8'] = create_conv_layer(prefix+'unet512_8', conv7, 64, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True) # 512 * 512
		conv9,vs['w9'],vs['b9'] = create_conv_layer(prefix+'unet512_9', conv8, 64, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True) # 512 * 512
	

	conv9_concat = tf.concat([conv9, conv1],3)

	conv10,vs['w10'],vs['b10'] = create_conv_layer(prefix+'unet512_10', conv9_concat, 64, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation) # 512 * 512
	conv11,vs['w11'],vs['b11'] = create_conv_layer(prefix+'unet512_11', conv10, 32, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation) # 512 * 512
	conv12,vs['w12'],vs['b12'] = create_conv_layer(prefix+'unet512_12', conv11, 8, 2, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation) # 512 * 512
	
	return conv12, vs



def run_unet512_12_V1(inputs, weights, inputdim = 3, prefix = "none_",last_activation = 'linear',deconv = False):
	#print(weights['w1'], weights['b1'])
	conv0= forward_conv_layer(prefix+'unet512_0', inputs, inputdim, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w0'], b=weights['b0'])
	conv1= forward_conv_layer(prefix+'unet512_1', conv0, 32, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w1'], b=weights['b1'])
	#print(conv1)
	conv2= forward_conv_layer(prefix+'unet512_2', conv1, 32, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w2'], b=weights['b2'])
	conv3= forward_conv_layer(prefix+'unet512_3', conv2, 64, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w3'], b=weights['b3']) # 128 * 128
	conv4= forward_conv_layer(prefix+'unet512_4', conv3, 64, 128, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w4'], b=weights['b4']) # 64 * 64

	#conv4_up = tf.image.resize_images(conv4, [128, 128], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	
	if deconv == False:
		conv4_up = tf.image.resize_nearest_neighbor(conv4, [128, 128])
		#conv4_up = imageUpSampling2x(conv4)

		conv5= forward_conv_layer(prefix+'unet512_5', conv4_up, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w5'], b=weights['b5']) # 128 * 128
	else:
		conv5= forward_conv_layer(prefix+'unet512_5', conv4, 128, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w5'], b=weights['b5'], deconv = True) # 128 * 128
	

	conv5_concat = tf.concat([conv5 , conv3],3)


	conv6= forward_conv_layer(prefix+'unet512_6', conv5_concat, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w6'], b=weights['b6']) # 128 * 128
	conv7= forward_conv_layer(prefix+'unet512_7', conv6, 64, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w7'], b=weights['b7']) # 128 * 128

	if deconv == False:
		conv7_up = tf.image.resize_images(conv7, [256, 256], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		#conv7_up = imageUpSampling2x(conv7)

		conv8= forward_conv_layer(prefix+'unet512_8', conv7_up, 32, 16, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w8'], b=weights['b8']) # 512 * 512
		conv8_up = tf.image.resize_images(conv8, [512, 512], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		#conv8_up = imageUpSampling2x(conv8)

		conv9= forward_conv_layer(prefix+'unet512_9', conv8_up, 16, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w9'], b=weights['b9']) # 512 * 512
	
	else:
		conv8= forward_conv_layer(prefix+'unet512_8', conv7, 64, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w8'], b=weights['b8'], deconv=True) # 512 * 512
		conv9= forward_conv_layer(prefix+'unet512_9', conv8, 64, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w9'], b=weights['b9'], deconv=True) # 512 * 512
	

	conv9_concat = tf.concat([conv9, conv1],3)

	conv10= forward_conv_layer(prefix+'unet512_10', conv9_concat, 64, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation, w=weights['w10'], b=weights['b10']) # 512 * 512
	conv11= forward_conv_layer(prefix+'unet512_11', conv10, 32, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation, w=weights['w11'], b=weights['b11']) # 512 * 512
	conv12= forward_conv_layer(prefix+'unet512_12', conv11, 8, 2, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation, w=weights['w12'], b=weights['b12']) # 512 * 512
	
	return conv12

def build_unet512_12_V1_fixed(inputs, inputdim = 3, prefix = "none_",last_activation = 'linear', deconv = False):
	vs = {}
	conv0,vs['w0'],vs['b0'] = create_conv_layer(prefix+'unet512_0', inputs, inputdim, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 256 * 256
	conv1,vs['w1'],vs['b1'] = create_conv_layer(prefix+'unet512_1', conv0, 32, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 256 * 256
	conv2,vs['w2'],vs['b2'] = create_conv_layer(prefix+'unet512_2', conv1, 32, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5) # 128 * 128 
	conv3,vs['w3'],vs['b3'] = create_conv_layer(prefix+'unet512_3', conv2, 64, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5) # 128 * 128
	conv4,vs['w4'],vs['b4'] = create_conv_layer(prefix+'unet512_4', conv3, 64, 128, stride_x = 2, stride_y = 2, kx = 5, ky = 5) # 64 * 64

	#conv4_up = tf.image.resize_images(conv4, [128, 128], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	
	conv5,vs['w5'],vs['b5'] = create_conv_layer(prefix+'unet512_5', conv4, 128, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, deconv = True) # 128 * 128
	


	conv5_concat = tf.concat([conv5 , conv3],3)


	conv6,vs['w6'],vs['b6'] = create_conv_layer(prefix+'unet512_6', conv5_concat, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128
	conv7,vs['w7'],vs['b7'] = create_conv_layer(prefix+'unet512_7', conv6, 64, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128

	
	conv8,vs['w8'],vs['b8'] = create_conv_layer(prefix+'unet512_8', conv7, 64, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True) # 512 * 512
	conv9,vs['w9'],vs['b9'] = create_conv_layer(prefix+'unet512_9', conv8, 64, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True) # 512 * 512
	

	conv9_concat = tf.concat([conv9, conv1],3)

	conv10,vs['w10'],vs['b10'] = create_conv_layer(prefix+'unet512_10', conv9_concat, 64, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 512 * 512
	conv11,vs['w11'],vs['b11'] = create_conv_layer(prefix+'unet512_11', conv10, 32, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 512 * 512
	conv12,vs['w12'],vs['b12'] = create_conv_layer(prefix+'unet512_12', conv11, 8, 2, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation) # 512 * 512
	
	return conv12, vs

def run_unet512_12_V1_fixed(inputs, weights, inputdim = 3, prefix = "none_",last_activation = 'linear',deconv = False):
	#print(weights['w1'], weights['b1'])
	conv0= forward_conv_layer(prefix+'unet512_0', inputs, inputdim, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w0'], b=weights['b0'])
	conv1= forward_conv_layer(prefix+'unet512_1', conv0, 32, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w1'], b=weights['b1'])
	#print(conv1)
	conv2= forward_conv_layer(prefix+'unet512_2', conv1, 32, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w2'], b=weights['b2'])
	conv3= forward_conv_layer(prefix+'unet512_3', conv2, 64, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w3'], b=weights['b3']) # 128 * 128
	conv4= forward_conv_layer(prefix+'unet512_4', conv3, 64, 128, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w4'], b=weights['b4']) # 64 * 64

	#conv4_up = tf.image.resize_images(conv4, [128, 128], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	
	
	conv5= forward_conv_layer(prefix+'unet512_5', conv4, 128, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w5'], b=weights['b5'], deconv = True) # 128 * 128
	

	conv5_concat = tf.concat([conv5 , conv3],3)


	conv6= forward_conv_layer(prefix+'unet512_6', conv5_concat, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w6'], b=weights['b6']) # 128 * 128
	conv7= forward_conv_layer(prefix+'unet512_7', conv6, 64, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w7'], b=weights['b7']) # 128 * 128

	
	conv8= forward_conv_layer(prefix+'unet512_8', conv7, 64, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w8'], b=weights['b8'], deconv=True) # 512 * 512
	conv9= forward_conv_layer(prefix+'unet512_9', conv8, 64, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w9'], b=weights['b9'], deconv=True) # 512 * 512
	

	conv9_concat = tf.concat([conv9, conv1],3)

	conv10= forward_conv_layer(prefix+'unet512_10', conv9_concat, 64, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w10'], b=weights['b10']) # 512 * 512
	conv11= forward_conv_layer(prefix+'unet512_11', conv10, 32, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w11'], b=weights['b11']) # 512 * 512
	conv12= forward_conv_layer(prefix+'unet512_12', conv11, 8, 2, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation, w=weights['w12'], b=weights['b12']) # 512 * 512
	
	return conv12

def build_simple_net_512_4_V1(inputs, inputdim = 3, prefix = "none_",last_activation = 'linear', deconv = False):
	vs = {}
	conv0,vs['w0'],vs['b0'] = create_conv_layer(prefix+'unet512_0', inputs, inputdim, 8, stride_x = 2, stride_y = 2, kx = 7, ky = 7) # 256 * 256
	conv1,vs['w1'],vs['b1'] = create_conv_layer(prefix+'unet512_1', conv0, 8, 16, stride_x = 2, stride_y = 2, kx = 7, ky = 7) # 256 * 256
	conv2,vs['w2'],vs['b2'] = create_conv_layer(prefix+'unet512_2', conv1, 16, 8, stride_x = 2, stride_y = 2, kx = 7, ky = 7,deconv=True) # 512 * 512
	conv3,vs['w3'],vs['b3'] = create_conv_layer(prefix+'unet512_3', conv2, 8, 8, stride_x = 2, stride_y = 2, kx = 7, ky = 7,deconv=True) # 512 * 512
	conv3 = tf.nn.dropout(conv3, keep_prob=0.6)
	conv4,vs['w4'],vs['b4'] = create_conv_layer(prefix+'unet512_4', conv3, 8, 2, activation = last_activation, stride_x = 1, stride_y = 1, kx = 7, ky = 7) # 512 * 512
		

	return conv4, vs


def build_unet512_12_V0(inputs, inputdim = 3, prefix = "none_",last_activation = 'linear', deconv = False):
	vs = {}
	conv0,vs['w0'],vs['b0'] = create_conv_layer(prefix+'unet512_0', inputs, inputdim, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 256 * 256
	conv1,vs['w1'],vs['b1'] = create_conv_layer(prefix+'unet512_1', conv0, 8, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 256 * 256
	conv2,vs['w2'],vs['b2'] = create_conv_layer(prefix+'unet512_2', conv1, 8, 16, stride_x = 2, stride_y = 2, kx = 5, ky = 5) # 128 * 128 
	conv3,vs['w3'],vs['b3'] = create_conv_layer(prefix+'unet512_3', conv2, 16, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5) # 128 * 128
	conv4,vs['w4'],vs['b4'] = create_conv_layer(prefix+'unet512_4', conv3, 32, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5) # 64 * 64

	#conv4_up = tf.image.resize_images(conv4, [128, 128], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	if deconv == False:
		conv4_up = tf.image.resize_nearest_neighbor(conv4, [128, 128])
		#conv4_up = imageUpSampling2x(conv4)
		conv5,vs['w5'],vs['b5'] = create_conv_layer(prefix+'unet512_5', conv4_up, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128
	else:
		conv5,vs['w5'],vs['b5'] = create_conv_layer(prefix+'unet512_5', conv4, 64, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5, deconv = True) # 128 * 128
	


	conv5_concat = tf.concat([conv5 , conv3],3)


	conv6,vs['w6'],vs['b6'] = create_conv_layer(prefix+'unet512_6', conv5_concat, 64, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128
	conv7,vs['w7'],vs['b7'] = create_conv_layer(prefix+'unet512_7', conv6, 32, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128

	if deconv == False:
		conv7_up = tf.image.resize_images(conv7, [256, 256], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		#conv7_up = imageUpSampling2x(conv7)
		conv8,vs['w8'],vs['b8'] = create_conv_layer(prefix+'unet512_8', conv7_up, 32, 16, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 512 * 512
		conv8_up = tf.image.resize_images(conv8, [512, 512], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		#conv8_up = imageUpSampling2x(conv8)
		conv9,vs['w9'],vs['b9'] = create_conv_layer(prefix+'unet512_9', conv8_up, 16, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 512 * 512
	else:
		conv8,vs['w8'],vs['b8'] = create_conv_layer(prefix+'unet512_8', conv7, 32, 16, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True) # 512 * 512
		conv9,vs['w9'],vs['b9'] = create_conv_layer(prefix+'unet512_9', conv8, 16, 8, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True) # 512 * 512
	

	conv9_concat = tf.concat([conv9, conv1],3)

	conv10,vs['w10'],vs['b10'] = create_conv_layer(prefix+'unet512_10', conv9_concat, 16, 16, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation) # 512 * 512
	conv11,vs['w11'],vs['b11'] = create_conv_layer(prefix+'unet512_11', conv10, 16, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation) # 512 * 512
	conv12,vs['w12'],vs['b12'] = create_conv_layer(prefix+'unet512_12', conv11, 8, 2, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation) # 512 * 512
	
	return conv12, vs



def run_unet512_12_V0(inputs, weights, inputdim = 3, prefix = "none_",last_activation = 'linear',deconv = False):
	#print(weights['w1'], weights['b1'])
	conv0= forward_conv_layer(prefix+'unet512_0', inputs, inputdim, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w0'], b=weights['b0'])
	conv1= forward_conv_layer(prefix+'unet512_1', conv0, 8, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w1'], b=weights['b1'])
	#print(conv1)
	conv2= forward_conv_layer(prefix+'unet512_2', conv1, 8, 16, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w2'], b=weights['b2'])
	conv3= forward_conv_layer(prefix+'unet512_3', conv2, 16, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w3'], b=weights['b3']) # 128 * 128
	conv4= forward_conv_layer(prefix+'unet512_4', conv3, 32, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w4'], b=weights['b4']) # 64 * 64

	#conv4_up = tf.image.resize_images(conv4, [128, 128], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	
	if deconv == False:
		conv4_up = tf.image.resize_nearest_neighbor(conv4, [128, 128])
		#conv4_up = imageUpSampling2x(conv4)

		conv5= forward_conv_layer(prefix+'unet512_5', conv4_up, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w5'], b=weights['b5']) # 128 * 128
	else:
		conv5= forward_conv_layer(prefix+'unet512_5', conv4, 64, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w5'], b=weights['b5'], deconv = True) # 128 * 128
	

	conv5_concat = tf.concat([conv5 , conv3],3)


	conv6= forward_conv_layer(prefix+'unet512_6', conv5_concat, 64, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w6'], b=weights['b6']) # 128 * 128
	conv7= forward_conv_layer(prefix+'unet512_7', conv6, 32, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w7'], b=weights['b7']) # 128 * 128

	if deconv == False:
		conv7_up = tf.image.resize_images(conv7, [256, 256], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		#conv7_up = imageUpSampling2x(conv7)

		conv8= forward_conv_layer(prefix+'unet512_8', conv7_up, 32, 16, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w8'], b=weights['b8']) # 512 * 512
		conv8_up = tf.image.resize_images(conv8, [512, 512], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		#conv8_up = imageUpSampling2x(conv8)

		conv9= forward_conv_layer(prefix+'unet512_9', conv8_up, 16, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5, w=weights['w9'], b=weights['b9']) # 512 * 512
	
	else:
		conv8= forward_conv_layer(prefix+'unet512_8', conv7, 32, 16, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w8'], b=weights['b8'], deconv=True) # 512 * 512
		conv9= forward_conv_layer(prefix+'unet512_9', conv8, 16, 8, stride_x = 2, stride_y = 2, kx = 5, ky = 5, w=weights['w9'], b=weights['b9'], deconv=True) # 512 * 512
	

	conv9_concat = tf.concat([conv9, conv1],3)

	conv10= forward_conv_layer(prefix+'unet512_10', conv9_concat, 16, 16, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation, w=weights['w10'], b=weights['b10']) # 512 * 512
	conv11= forward_conv_layer(prefix+'unet512_11', conv10, 16, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation, w=weights['w11'], b=weights['b11']) # 512 * 512
	conv12= forward_conv_layer(prefix+'unet512_12', conv11, 8, 2, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation, w=weights['w12'], b=weights['b12']) # 512 * 512
	
	return conv12


# def build_unet512_IntraClass_8(inputs, inputdim = 3,is_training=False, prefix = "none_",last_activation = 'linear',deconv=True):
# 	vs = {}
# 	use_batchnorm = False
# 	conv0,vs['w0'],vs['b0'] = create_conv_layer(prefix+'unet512_0', inputs, inputdim, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,batchnorm=use_batchnorm, is_training=is_training) # 256 * 256
# 	conv1,vs['w1'],vs['b1'] = create_conv_layer(prefix+'unet512_1', conv0, 32, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,batchnorm=use_batchnorm, is_training=is_training) # 128 * 128
# 	conv2,vs['w2'],vs['b2'] = create_conv_layer(prefix+'unet512_2', conv1, 32, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,batchnorm=use_batchnorm, is_training=is_training) # 64 * 64 
	
# 	# Maybe add a bottleneck link here

# 	conv3,vs['w3'],vs['b3'] = create_conv_layer(prefix+'unet512_3', conv2, 32, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True,batchnorm=use_batchnorm, is_training=is_training) # 128 * 128
# 	conv4,vs['w4'],vs['b4'] = create_conv_layer(prefix+'unet512_4', conv3, 32, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True,batchnorm=use_batchnorm, is_training=is_training) # 256 * 256
# 	conv5,vs['w5'],vs['b5'] = create_conv_layer(prefix+'unet512_5', conv4, 32, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True,batchnorm=use_batchnorm, is_training=is_training) # 512 * 512
	
# 	conv5_concat = tf.concat([conv5 , inputs],3)
# 	conv6,vs['w6'],vs['b6'] = create_conv_layer(prefix+'unet512_6', conv5_concat, 32+inputdim, 32, stride_x = 1, stride_y = 1, kx = 7, ky = 7,batchnorm=use_batchnorm, is_training=is_training) # 512*512
# 	conv7,vs['w7'],vs['b7'] = create_conv_layer(prefix+'unet512_7', conv6, 32, 16, stride_x = 1, stride_y = 1, kx = 7, ky = 7,batchnorm=use_batchnorm, is_training=is_training) # 512*512
# 	conv8,vs['w8'],vs['b8'] = create_conv_layer(prefix+'unet512_8', conv7, 16, 2, stride_x = 1, stride_y = 1, kx = 7, ky = 7,activation = last_activation,batchnorm=use_batchnorm, is_training=is_training) # 512*512
	

# 	print("conv8",conv8)
# 	return conv8, vs

# def run_unet512_IntraClass_8(inputs, weights, inputdim = 3, is_training=False, prefix = "none_",last_activation = 'linear',deconv=True):
# 	use_batchnorm = False

# 	conv0 = forward_conv_layer(prefix+'unet512_0', inputs, inputdim, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,w=weights['w0'], b=weights['b0'],batchnorm=use_batchnorm, is_training=is_training) # 256 * 256
# 	conv1 = forward_conv_layer(prefix+'unet512_1', conv0, 32, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,w=weights['w1'], b=weights['b1'],batchnorm=use_batchnorm, is_training=is_training) # 128 * 128
# 	conv2 = forward_conv_layer(prefix+'unet512_2', conv1, 32, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,w=weights['w2'], b=weights['b2'],batchnorm=use_batchnorm, is_training=is_training) # 64 * 64 
	
# 	# Maybe add a bottleneck link here

# 	conv3 = forward_conv_layer(prefix+'unet512_3', conv2, 32, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True,w=weights['w3'], b=weights['b3'],batchnorm=use_batchnorm, is_training=is_training) # 128 * 128
# 	conv4 = forward_conv_layer(prefix+'unet512_4', conv3, 32, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True,w=weights['w4'], b=weights['b4'],batchnorm=use_batchnorm, is_training=is_training) # 256 * 256
# 	conv5 = forward_conv_layer(prefix+'unet512_5', conv4, 32, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True,w=weights['w5'], b=weights['b5'],batchnorm=use_batchnorm, is_training=is_training) # 512 * 512
	
# 	conv5_concat = tf.concat([conv5 , inputs],3)
# 	conv6 = forward_conv_layer(prefix+'unet512_6', conv5_concat, 32+inputdim, 32, stride_x = 1, stride_y = 1, kx = 7, ky = 7,w=weights['w6'], b=weights['b6'],batchnorm=use_batchnorm, is_training=is_training) # 512*512
# 	conv7 = forward_conv_layer(prefix+'unet512_7', conv6, 32, 16, stride_x = 1, stride_y = 1, kx = 7, ky = 7,w=weights['w7'], b=weights['b7'],batchnorm=use_batchnorm, is_training=is_training) # 512*512
# 	conv8 = forward_conv_layer(prefix+'unet512_8', conv7, 16, 2, stride_x = 1, stride_y = 1, kx = 7, ky = 7,activation = last_activation,w=weights['w8'], b=weights['b8'],batchnorm=use_batchnorm, is_training=is_training) # 512*512
	


# 	return conv8

def build_unet512_IntraClass_8(inputs, inputdim = 3,is_training=False, prefix = "none_",last_activation = 'linear',deconv=True):
	vs = {}
	use_batchnorm = False
	conv0,vs['w0'],vs['b0'] = create_conv_layer(prefix+'unet512_0', inputs, inputdim, 8, stride_x = 2, stride_y = 2, kx = 5, ky = 5,batchnorm=use_batchnorm, is_training=is_training) # 256 * 256
	conv1,vs['w1'],vs['b1'] = create_conv_layer(prefix+'unet512_1', conv0, 8, 16, stride_x = 2, stride_y = 2, kx = 5, ky = 5,batchnorm=use_batchnorm, is_training=is_training) # 128 * 128
	conv2,vs['w2'],vs['b2'] = create_conv_layer(prefix+'unet512_2', conv1, 16, 32, stride_x = 2, stride_y = 2, kx = 7, ky = 7,batchnorm=use_batchnorm, is_training=is_training) # 64 * 64 
	
	# Maybe add a bottleneck link here

	conv3,vs['w3'],vs['b3'] = create_conv_layer(prefix+'unet512_3', conv2, 32, 16, stride_x = 2, stride_y = 2, kx = 7, ky = 7,deconv=True,batchnorm=use_batchnorm, is_training=is_training) # 128 * 128
	conv4,vs['w4'],vs['b4'] = create_conv_layer(prefix+'unet512_4', conv3, 16, 8, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True,batchnorm=use_batchnorm, is_training=is_training) # 256 * 256
	conv5,vs['w5'],vs['b5'] = create_conv_layer(prefix+'unet512_5', conv4, 8, 8, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True,batchnorm=use_batchnorm, is_training=is_training) # 512 * 512
	
	conv5_concat = tf.concat([conv5 , inputs],3)
	conv6,vs['w6'],vs['b6'] = create_conv_layer(prefix+'unet512_6', conv5_concat, 8+inputdim, 8+inputdim, stride_x = 1, stride_y = 1, kx = 5, ky = 5,batchnorm=use_batchnorm, is_training=is_training) # 512*512
	conv7,vs['w7'],vs['b7'] = create_conv_layer(prefix+'unet512_7', conv6, 8+inputdim, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5,batchnorm=use_batchnorm, is_training=is_training) # 512*512
	conv8,vs['w8'],vs['b8'] = create_conv_layer(prefix+'unet512_8', conv7, 8, 2, stride_x = 1, stride_y = 1, kx = 3, ky = 3,activation = last_activation,batchnorm=use_batchnorm, is_training=is_training) # 512*512
	

	print("conv8",conv8)
	return conv8, vs

def run_unet512_IntraClass_8(inputs, weights, inputdim = 3, is_training=False, prefix = "none_",last_activation = 'linear',deconv=True):
	use_batchnorm = False

	conv0 = forward_conv_layer(prefix+'unet512_0', inputs, inputdim, 8, stride_x = 2, stride_y = 2, kx = 5, ky = 5,w=weights['w0'], b=weights['b0'],batchnorm=use_batchnorm, is_training=is_training) # 256 * 256
	conv1 = forward_conv_layer(prefix+'unet512_1', conv0, 8, 16, stride_x = 2, stride_y = 2, kx = 5, ky = 5,w=weights['w1'], b=weights['b1'],batchnorm=use_batchnorm, is_training=is_training) # 128 * 128
	conv2 = forward_conv_layer(prefix+'unet512_2', conv1, 16, 32, stride_x = 2, stride_y = 2, kx = 7, ky = 7,w=weights['w2'], b=weights['b2'],batchnorm=use_batchnorm, is_training=is_training) # 64 * 64 
	
	# Maybe add a bottleneck link here

	conv3 = forward_conv_layer(prefix+'unet512_3', conv2, 32, 16, stride_x = 2, stride_y = 2, kx = 7, ky = 7,deconv=True,w=weights['w3'], b=weights['b3'],batchnorm=use_batchnorm, is_training=is_training) # 128 * 128
	conv4 = forward_conv_layer(prefix+'unet512_4', conv3, 16, 8, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True,w=weights['w4'], b=weights['b4'],batchnorm=use_batchnorm, is_training=is_training) # 256 * 256
	conv5 = forward_conv_layer(prefix+'unet512_5', conv4, 8, 8, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True,w=weights['w5'], b=weights['b5'],batchnorm=use_batchnorm, is_training=is_training) # 512 * 512
	
	conv5_concat = tf.concat([conv5 , inputs],3)
	conv6 = forward_conv_layer(prefix+'unet512_6', conv5_concat, 8+inputdim, 8+inputdim, stride_x = 1, stride_y = 1, kx = 5, ky = 5,w=weights['w6'], b=weights['b6'],batchnorm=use_batchnorm, is_training=is_training) # 512*512
	conv7 = forward_conv_layer(prefix+'unet512_7', conv6, 8+inputdim, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5,w=weights['w7'], b=weights['b7'],batchnorm=use_batchnorm, is_training=is_training) # 512*512
	conv8 = forward_conv_layer(prefix+'unet512_8', conv7, 8, 2, stride_x = 1, stride_y = 1, kx = 3, ky = 3,activation = last_activation,w=weights['w8'], b=weights['b8'],batchnorm=use_batchnorm, is_training=is_training) # 512*512
	

	return conv8





def L2Loss(a,b):
	return tf.reduce_mean(tf.multiply(tf.square(a-b),b+0.1)) 

def CrossEntropy(a,b):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.concat([b,1-b],3), logits=a))
	#return tf.reduce_mean(tf.multiply(tf.square(a-b),b+0.1)) 




# only tune the last few layers...   a small U-net + 3 more layers...  only update the last 3 layers
def buildMetaBlockV1(inputA, outputA, inputB, outputB, loss_func = CrossEntropy, inner_lr = 0.01, inner_step = 5, stopgrad = True, decay = 0.9, layer_st = 10, layer_ed = 13, build_cnnmodel=None, run_cnnmodel=None, inputdim=3):

	#build_cnnmodel = build_unet512_10
	#run_cnnmodel = run_unet512_10

	#build_cnnmodel = build_unet512_12_V1
	#run_cnnmodel = run_unet512_12_V1

	if build_cnnmodel is None:
		build_cnnmodel = build_unet512_12_V1_fixed
		run_cnnmodel = run_unet512_12_V1_fixed


	task_losses = []
	task_outputs = []

	groupA_losses = []

	inner_output, weights = build_cnnmodel(inputA, prefix="first_", deconv=True,inputdim=inputdim)

	subset_weights = {}
	for l in xrange(layer_st,layer_ed):
		subset_weights['w'+str(l)] = weights['w'+str(l)]
		subset_weights['b'+str(l)] = weights['b'+str(l)]

	inner_loss = loss_func(inner_output, outputA)
	task_losses.append(inner_loss)
	groupA_losses.append(inner_loss)

	grads = tf.gradients(inner_loss, list(subset_weights.values()))
	if stopgrad:
		grads = [tf.stop_gradient(grad) for grad in grads]

	max_grad = 0
	for g in grads:
		max_grad = tf.maximum(tf.reduce_max(tf.abs(g)), max_grad)


	gradients = dict(zip(subset_weights.keys(), grads))
	sub_fast_weights = dict(zip(subset_weights.keys(), [subset_weights[key] - inner_lr*gradients[key] for key in subset_weights.keys()]))

	fast_weights = {}
	for k in weights.keys():
		if k in subset_weights:
			fast_weights[k] = sub_fast_weights[k]
		else:
			fast_weights[k] = weights[k]


	output1 = run_cnnmodel(inputB, fast_weights,inputdim=inputdim)
	#print(output1)
	task_outputs.append(tf.nn.softmax(output1))
	loss1 = loss_func(output1, outputB)
	task_losses.append(loss1)

	for j in xrange(1,inner_step):
		loss = loss_func(run_cnnmodel(inputA, fast_weights,inputdim=inputdim), outputA)
		groupA_losses.append(loss)
		grads = tf.gradients(loss, list(sub_fast_weights.values()))
		if stopgrad:
			grads = [tf.stop_gradient(grad) for grad in grads]


		gradients = dict(zip(sub_fast_weights.keys(), grads))
		inner_lr = inner_lr * decay
		sub_fast_weights = dict(zip(sub_fast_weights.keys(), [sub_fast_weights[key] - inner_lr*gradients[key] for key in sub_fast_weights.keys()]))
		
		fast_weights = {}
		for k in weights.keys():
			if k in subset_weights:
				fast_weights[k] = sub_fast_weights[k]
			else:
				fast_weights[k] = weights[k]


		output1 = run_cnnmodel(inputB, fast_weights,inputdim=inputdim)
		loss1 = loss_func(output1, outputB)
		task_losses.append(loss1)
		task_outputs.append(tf.nn.softmax(output1))


	return task_losses, task_outputs, inner_output, loss, groupA_losses, max_grad

def buildMetaBlockV1_old(inputA, outputA, inputB, outputB, loss_func = CrossEntropy, inner_lr = 0.01, inner_step = 5, stopgrad = True, decay = 0.9, layer_st = 10, layer_ed = 13, build_cnn_model = build_unet512_12_V1_fixed, run_cnn_model = run_unet512_12_V1_fixed, run_cnnmodel=None, inputdim=3):

	#build_cnnmodel = build_unet512_10
	#run_cnnmodel = run_unet512_10

	#build_cnnmodel = build_unet512_12_V1
	#run_cnnmodel = run_unet512_12_V1

	# if build_cnnmodel is None:
	# 	build_cnnmodel = build_unet512_12_V1_fixed
	# 	run_cnnmodel = run_unet512_12_V1_fixed


	build_cnnmodel = build_cnn_model 
	run_cnnmodel = run_cnn_model

	task_losses = []
	task_outputs = []

	groupA_losses = []

	inner_output, weights = build_cnnmodel(inputA, prefix="first_", deconv=True,inputdim=inputdim)

	subset_weights = {}
	for l in xrange(layer_st,layer_ed):
		subset_weights['w'+str(l)] = weights['w'+str(l)]
		subset_weights['b'+str(l)] = weights['b'+str(l)]

	inner_loss = loss_func(inner_output, outputA)
	task_losses.append(inner_loss)
	groupA_losses.append(inner_loss)

	grads = tf.gradients(inner_loss, list(subset_weights.values()))
	if stopgrad:
		grads = [tf.stop_gradient(grad) for grad in grads]

	
	gradients = dict(zip(subset_weights.keys(), grads))
	sub_fast_weights = dict(zip(subset_weights.keys(), [subset_weights[key] - inner_lr*gradients[key] for key in subset_weights.keys()]))

	fast_weights = {}
	for k in weights.keys():
		if k in subset_weights:
			fast_weights[k] = sub_fast_weights[k]
		else:
			fast_weights[k] = weights[k]


	output1 = run_cnnmodel(inputB, fast_weights,inputdim=inputdim)
	#print(output1)
	task_outputs.append(tf.nn.softmax(output1))
	loss1 = loss_func(output1, outputB)
	task_losses.append(loss1)

	for j in xrange(1,inner_step):
		loss = loss_func(run_cnnmodel(inputA, fast_weights,inputdim=inputdim), outputA)

		grads = tf.gradients(loss, list(sub_fast_weights.values()))
		if stopgrad:
			grads = [tf.stop_gradient(grad) for grad in grads]


		gradients = dict(zip(sub_fast_weights.keys(), grads))
		inner_lr = inner_lr * decay
		sub_fast_weights = dict(zip(sub_fast_weights.keys(), [sub_fast_weights[key] - inner_lr*gradients[key] for key in sub_fast_weights.keys()]))
		
		fast_weights = {}
		for k in weights.keys():
			if k in subset_weights:
				fast_weights[k] = sub_fast_weights[k]
			else:
				fast_weights[k] = weights[k]


		output1 = run_cnnmodel(inputB, fast_weights,inputdim=inputdim)
		loss1 = loss_func(output1, outputB)
		task_losses.append(loss1)
		task_outputs.append(tf.nn.softmax(output1))


	return task_losses, task_outputs, inner_output, loss


# PosNegNet  --> two outputs
# Different training mode
# (1) CNN Architecture 
#	- Class Level Segmentation
#   - InterClass Segmentation 


# def build_PosNegNet_V1(inputs, inputdim = 3, prefix = "none_",last_activation = 'linear'):
# 	vs = {}

# 	conv0,vs['w0'],vs['b0'] = create_conv_layer(prefix+'unet512_0', inputs, inputdim, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 256 * 256
# 	conv1,vs['w1'],vs['b1'] = create_conv_layer(prefix+'unet512_1', conv0, 32, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 256 * 256
# 	conv2,vs['w2'],vs['b2'] = create_conv_layer(prefix+'unet512_2', conv1, 32, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5) # 128 * 128 
# 	conv3,vs['w3'],vs['b3'] = create_conv_layer(prefix+'unet512_3', conv2, 64, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5) # 128 * 128
# 	conv4,vs['w4'],vs['b4'] = create_conv_layer(prefix+'unet512_4', conv3, 64, 128, stride_x = 2, stride_y = 2, kx = 5, ky = 5) # 64 * 64

# 	#conv4_up = tf.image.resize_images(conv4, [128, 128], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# 	conv5,vs['w5'],vs['b5'] = create_conv_layer(prefix+'unet512_5', conv4, 128, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5, deconv = True) # 128 * 128
# 	conv5_concat = tf.concat([conv5 , conv3],3)

# 	conv6,vs['w6'],vs['b6'] = create_conv_layer(prefix+'unet512_6', conv5_concat, 128, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128
# 	conv7,vs['w7'],vs['b7'] = create_conv_layer(prefix+'unet512_7', conv6, 64, 64, stride_x = 1, stride_y = 1, kx = 5, ky = 5) # 128 * 128

# 	conv8,vs['w8'],vs['b8'] = create_conv_layer(prefix+'unet512_8', conv7, 64, 64, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True) # 512 * 512
# 	conv9,vs['w9'],vs['b9'] = create_conv_layer(prefix+'unet512_9', conv8, 64, 32, stride_x = 2, stride_y = 2, kx = 5, ky = 5,deconv=True) # 512 * 512
	
# 	conv9_concat = tf.concat([conv9, conv1],3)

# 	conv10,vs['w10'],vs['b10'] = create_conv_layer(prefix+'unet512_10', conv9_concat, 64, 32, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation) # 512 * 512
# 	conv11,vs['w11'],vs['b11'] = create_conv_layer(prefix+'unet512_11', conv10, 32, 8, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation) # 512 * 512
# 	conv12,vs['w12'],vs['b12'] = create_conv_layer(prefix+'unet512_12', conv11, 8, 2, stride_x = 1, stride_y = 1, kx = 5, ky = 5, activation = last_activation) # 512 * 512
	
# 	return conv12, vs




# def buildPosNegNetV1(inputA, outputA, inputB, outputB, loss_func = CrossEntropy, inner_lr = 0.01, inner_step = 5, stopgrad = True, decay = 0.9, layer_st = 10, layer_ed = 13):

# 	#build_cnnmodel = build_unet512_10
# 	#run_cnnmodel = run_unet512_10

# 	build_cnnmodel = build_unet512_12_V1
# 	run_cnnmodel = run_unet512_12_V1


# 	task_losses = []
# 	task_outputs = []

# 	inner_output, weights = build_cnnmodel(inputA, prefix="first_", deconv=True)

# 	subset_weights = {}
# 	for l in xrange(layer_st,layer_ed):
# 		subset_weights['w'+str(l)] = weights['w'+str(l)]
# 		subset_weights['b'+str(l)] = weights['b'+str(l)]

# 	inner_loss = loss_func(inner_output, outputA)
# 	task_losses.append(inner_loss)

# 	grads = tf.gradients(inner_loss, list(subset_weights.values()))
# 	if stopgrad:
# 		grads = [tf.stop_gradient(grad) for grad in grads]

# 	gradients = dict(zip(subset_weights.keys(), grads))
# 	sub_fast_weights = dict(zip(subset_weights.keys(), [subset_weights[key] - inner_lr*gradients[key] for key in subset_weights.keys()]))

# 	fast_weights = {}
# 	for k in weights.keys():
# 		if k in subset_weights:
# 			fast_weights[k] = sub_fast_weights[k]
# 		else:
# 			fast_weights[k] = weights[k]


# 	output1 = run_cnnmodel(inputB, fast_weights,deconv=True)
# 	#print(output1)
# 	task_outputs.append(tf.nn.softmax(output1))
# 	loss1 = loss_func(output1, outputB)
# 	task_losses.append(loss1)

# 	for j in xrange(1,inner_step):
# 		loss = loss_func(run_cnnmodel(inputA, fast_weights,deconv=True), outputA)
# 		grads = tf.gradients(loss, list(sub_fast_weights.values()))
# 		if stopgrad:
# 			grads = [tf.stop_gradient(grad) for grad in grads]

# 		gradients = dict(zip(sub_fast_weights.keys(), grads))
# 		inner_lr = inner_lr * decay
# 		sub_fast_weights = dict(zip(sub_fast_weights.keys(), [sub_fast_weights[key] - inner_lr*gradients[key] for key in sub_fast_weights.keys()]))
		
# 		fast_weights = {}
# 		for k in weights.keys():
# 			if k in subset_weights:
# 				fast_weights[k] = sub_fast_weights[k]
# 			else:
# 				fast_weights[k] = weights[k]


# 		output1 = run_cnnmodel(inputB, fast_weights,deconv=True)
# 		loss1 = loss_func(output1, outputB)
# 		task_losses.append(loss1)
# 		task_outputs.append(tf.nn.softmax(output1))


# 	return task_losses, task_outputs, inner_output









