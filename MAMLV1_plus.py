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
import CNNmodel

image_size = 512

# Meta Examples Num during training 5 (high quality examples)
# Learning Step 10?

def create_conv_layer(name, input_tensor, in_channels, out_channels, is_training = True, activation='relu', kx = 3, ky = 3, stride_x = 2, stride_y = 2, batchnorm=False, padding='VALID', add=None, deconv = False):
	if deconv == False:
		input_tensor = tf.pad(input_tensor, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")


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
		input_tensor = tf.pad(input_tensor, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")

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

def imageUpSampling2x(tensor_in):
	batch = tf.shape(tensor_in)[0]
	dimx = tf.shape(tensor_in)[1]
	dimy = tf.shape(tensor_in)[2]
	channel_n = tf.shape(tensor_in)[3]

	#new_shape = tf.constant([batch, dimx*2, dimy*2, channel_n])
	tmp = tf.fill([batch, dimx*2, dimy*2, channel_n],0.0)

	tmp[:,::2,::2,:] = tensor_in[:,:,:,:]
	tmp[:,1::2,::2,:] = tensor_in[:,:,:,:]
	tmp[:,::2,1::2,:] = tensor_in[:,:,:,:]
	tmp[:,1::2,1::2,:] = tensor_in[:,:,:,:]

	return tmp


def lrelu(x, alpha):
	return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def sigmoid_weighted(color_intensity, color_central = 74, color_scale = 0.1):
	return tf.nn.sigmoid((color_intensity - color_central)*color_scale) * 5.0 


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



def L2Loss(a,b):
	return tf.reduce_mean(tf.multiply(tf.square(a-b),b+0.1)) 

def CrossEntropy(a,b):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.concat([b,1-b],3), logits=a))
	#return tf.reduce_mean(tf.multiply(tf.square(a-b),b+0.1)) 

def buildMetaBlock(inputA, outputA, inputB, outputB, loss_func = CrossEntropy, inner_lr = 0.01, inner_step = 5):

	#build_cnnmodel = build_unet512_10
	#run_cnnmodel = run_unet512_10

	build_cnnmodel = CNNmodel.build_unet512_12
	run_cnnmodel = CNNmodel.run_unet512_12


	task_losses = []
	task_outputs = []

	inner_output, weights = build_cnnmodel(inputA, prefix="first_", deconv=True)
	inner_loss = loss_func(inner_output, outputA)
	task_losses.append(inner_loss)

	grads = tf.gradients(inner_loss, list(weights.values()))
	#if inner_step > 1:
	#	grads = [tf.stop_gradient(grad) for grad in grads]

	gradients = dict(zip(weights.keys(), grads))
	fast_weights = dict(zip(weights.keys(), [weights[key] - inner_lr*gradients[key] for key in weights.keys()]))

	output1 = run_cnnmodel(inputB, fast_weights,deconv=True)
	#print(output1)
	task_outputs.append(tf.nn.softmax(output1))
	loss1 = loss_func(output1, outputB)
	task_losses.append(loss1)

	for j in xrange(1,inner_step):
		loss = loss_func(run_cnnmodel(inputA, fast_weights,deconv=True), outputA)
		grads = tf.gradients(loss, list(fast_weights.values()))
		#grads = [tf.stop_gradient(grad) for grad in grads]

		gradients = dict(zip(fast_weights.keys(), grads))
		fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - inner_lr*gradients[key] for key in fast_weights.keys()]))
		output1 = run_cnnmodel(inputB, fast_weights,deconv=True)
		loss1 = loss_func(output1, outputB)
		task_losses.append(loss1)
		task_outputs.append(tf.nn.softmax(output1))



	return task_losses, task_outputs, inner_output


class MAML(object):
	def __init__(self, sess, num_test_updates = 20, inner_lr = 0.001):
		self.num_updates = 10
		self.num_test_updates = 20
		self.num_test_updates = num_test_updates
		self.meta_lr_val = 0.001
		self.meta_lr_val = 0.0005

		self.meta_lr = tf.placeholder(tf.float32, shape=[])

		self.inner_lr = 0.002
		self.inner_lr_test = 0.002

		self.sess = sess


		self.inputA = tflearn.input_data(shape = [None, image_size, image_size, 3])
		self.outputA = tflearn.input_data(shape = [None, image_size, image_size, 1])
		self.inputB = tflearn.input_data(shape = [None, image_size, image_size, 3])
		self.outputB = tflearn.input_data(shape = [None, image_size, image_size, 1])

		with tf.variable_scope("foo", reuse=False):
			self.task_losses, self.task_outputs, _ = CNNmodel.buildMetaBlockV1(self.inputA, self.outputA, self.inputB, self.outputB, inner_step = self.num_updates, inner_lr = self.inner_lr, layer_st = 0, layer_ed=13)
		with tf.variable_scope("foo", reuse=True):
			self.task_losses_test, self.task_test_outputs, self.debug_inner_output = CNNmodel.buildMetaBlockV1(self.inputA, self.outputA, self.inputB, self.outputB, inner_step = self.num_test_updates, inner_lr = self.inner_lr_test, layer_st = 0, layer_ed=13)

		# optimizer = tf.train.AdamOptimizer(self.meta_lr)
		# self.gvs = gvs = optimizer.compute_gradients(self.task_losses[self.num_updates-1])
		# self.metatrain_op = optimizer.apply_gradients(gvs)


		self.metatrain_op = tf.train.AdamOptimizer(learning_rate=self.meta_lr).minimize(self.task_losses[self.num_updates-1])
		#self.metatrain_op = tf.train.AdamOptimizer(learning_rate=self.meta_lr).minimize(self.task_losses[self.num_updates-1] + (self.task_losses[self.num_updates-1] - self.task_losses[self.num_updates-2]))


		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep=100)


		# self.change_lr_op = self.meta_lr.assign(self.meta_lr_val)

		# self.sess.run([self.change_lr_op], feed_dict = {self.meta_lr_val:self.meta_lr_val})


	def trainModel(self, inputA, outputA, inputB, outputB, scale = 1.0):

		return self.sess.run([self.metatrain_op, self.task_losses[self.num_updates-1]] + self.task_losses, feed_dict = {self.inputA:inputA, self.outputA:outputA, self.inputB:inputB, self.outputB:outputB, self.meta_lr:self.meta_lr_val * scale})

	def runModel(self, inputA, outputA, inputB, outputB):
		return self.sess.run([self.task_test_outputs[self.num_test_updates-1], self.task_losses_test[self.num_test_updates-1], self.debug_inner_output], feed_dict = {self.inputA:inputA, self.outputA:outputA, self.inputB:inputB,self.outputB:outputB, self.meta_lr:self.meta_lr_val})

	def saveModel(self, path):
		self.saver.save(self.sess, path)

	def restoreModel(self, path):
		self.saver.restore(self.sess, path)

class DataLoader(object):
	def __init__(self, foldername, region_num, task_num, preload = False, dist = [0.2,0.2,0.2,0.2,0.2]):
		self.region_num = region_num 
		self.task_num = task_num
		self.region_size = 4096
		self.foldername = foldername
		self.preload = preload
		self.cc = 0
		self.interval = 100 # change task every 5 iterations
		self.total_time = 0


		self.distribution = [0.0] * task_num

		if preload == False:
			self.distribution = dist
			return 



		input_imgs = []
		target_imgs = []

		for i in xrange(region_num):
			target_imgs.append([])
			input_imgs.append(scipy.ndimage.imread(foldername + "/"+"region%d_sat.png"%i).astype(np.float)/255.0)
			for j in xrange(task_num):
				target_imgs[-1].append(scipy.ndimage.imread(foldername + "/"+"region%d_t%d.png"%(i,j+1)).astype(np.float)/255.0)

				self.distribution[j] += np.sum(target_imgs[-1][-1], axis =(0,1)) / (4096*4096)



			print("done with region", i, self.distribution)


		d_sum = 0
		for i in xrange(task_num):
			self.distribution[i] = np.sqrt(1.0/self.distribution[i])
			d_sum += self.distribution[i]

		for i in xrange(task_num):
			self.distribution[i] /= d_sum 

		print("Train Distribution", self.distribution)



		self.input_imgs = input_imgs 
		self.target_imgs = target_imgs

	def getBatch(self, sizeA=20, sizeB=20, update=True, allTask = False):
		ts = time()

		inputA = np.zeros((sizeA, image_size, image_size, 3))
		outputA = np.zeros((sizeA, image_size, image_size, 1))

		inputB = np.zeros((sizeB, image_size, image_size, 3))
		outputB = np.zeros((sizeB, image_size, image_size, 1))


		if self.cc % self.interval == 0:
			self.task_id = np.random.choice([0,1,2,3,4], p = self.distribution)
			#self.task_id = random.randint(0, self.task_num-1)


		self.cc += 1

		task_id = self.task_id

		#task_id = random.randint(0, self.task_num-1)

		# task_id = random.randint(0, 1)

		# if task_id == 0:
		# 	task_id = 1
		# else:
		# 	task_id = 3

		if self.preload == True:
			# Sample A
			for i in xrange(sizeA):
				while True:
					region_id = random.randint(0, self.region_num - 1)
					x = random.randint(0, self.region_size-image_size-1)
					y = random.randint(0, self.region_size-image_size-1)

					v_sum = np.sum(self.target_imgs[region_id][task_id][x:x+512,y:y+512])
					if v_sum > 512*512*0.05*0.05:
						break


				inputA[i, :,:,:] = self.input_imgs[region_id][x:x+512,y:y+512,:]
				outputA[i,:,:,0] = self.target_imgs[region_id][task_id][x:x+512,y:y+512]

				#	break

			# Sample B
			for i in xrange(sizeA):
				while True:
					region_id = random.randint(0, self.region_num - 1)
					x = random.randint(0, self.region_size-image_size-1)
					y = random.randint(0, self.region_size-image_size-1)

					v_sum = np.sum(self.target_imgs[region_id][task_id][x:x+512,y:y+512])
					if v_sum > 512*512*0.05*0.05:
						break

				inputB[i, :,:,:] = self.input_imgs[region_id][x:x+512,y:y+512,:]
				outputB[i,:,:,0] = self.target_imgs[region_id][task_id][x:x+512,y:y+512]

				#break
		else:
			# Sample A
			while True:
				if update == True:
					foldername = self.foldername
					region_id = random.randint(0, self.region_num - 1)
					self.input_imgs = scipy.ndimage.imread(foldername + "/"+"region%d_sat.png"%region_id).astype(np.float)/255.0
					self.target_imgs = []

					for j in xrange(self.task_num):
						self.target_imgs.append(scipy.ndimage.imread(foldername + "/"+"region%d_t%d.png"%(region_id,j+1)).astype(np.float)/255.0)

				retry_c = 0
				for i in xrange(sizeA):
					retry_c = 0

					while True:
						x = random.randint(0, self.region_size-image_size-1)
						y = random.randint(0, self.region_size-image_size-1)
						v_sum = np.sum(self.target_imgs[task_id][x:x+512,y:y+512])
						if v_sum > 512*512*0.05*0.05:
							break
						retry_c = retry_c + 1
						if retry_c > 1000:
							retry_c = -1
							break

					if retry_c == -1:
						break

					inputA[i, :,:,:] = self.input_imgs[x:x+512,y:y+512,:]
					outputA[i,:,:,0] = self.target_imgs[task_id][x:x+512,y:y+512]

					#	break
				if retry_c == -1:
					update = True
					continue

				# Sample B
				for i in xrange(sizeA):
					retry_c = 0
					while True:
						x = random.randint(0, self.region_size-image_size-1)
						y = random.randint(0, self.region_size-image_size-1)
						v_sum = np.sum(self.target_imgs[task_id][x:x+512,y:y+512])
						if v_sum > 512*512*0.05*0.05:
							break
						retry_c = retry_c + 1
						if retry_c > 10:
							retry_c = -1
							break

					if retry_c == -1:
						break

					inputB[i, :,:,:] = self.input_imgs[x:x+512,y:y+512,:]
					outputB[i,:,:,0] = self.target_imgs[task_id][x:x+512,y:y+512]

					#break
				if retry_c == -1:
					update = True
					continue

				break

		self.total_time += time() - ts 

		return inputA, outputA, inputB, outputB, task_id



	def getTestBatchs(self, sizeA=20, sizeB=20):

		inputAs = []
		inputBs = []
		outputAs = []
		outputBs = []

		for i in xrange(self.task_num):
			inputAs.append(np.zeros((sizeA, image_size, image_size, 3)))
			outputAs.append(np.zeros((sizeA, image_size, image_size, 1)))

			inputBs.append(np.zeros((sizeB, image_size, image_size, 3)))
			outputBs.append(np.zeros((sizeB, image_size, image_size, 1)))

		foldername = self.foldername
		region_id = random.randint(0, self.region_num - 1)
		self.input_imgs = scipy.ndimage.imread(foldername + "/"+"region%d_sat.png"%region_id).astype(np.float)/255.0
		self.target_imgs = []

		for j in xrange(self.task_num):
			self.target_imgs.append(scipy.ndimage.imread(foldername + "/"+"region%d_t%d.png"%(region_id,j+1)).astype(np.float)/255.0)



		for i in xrange(sizeA):
			#while True:
			x = random.randint(0, self.region_size-image_size-1)
			y = random.randint(0, self.region_size-image_size-1)
				
			for j in xrange(self.task_num):
				inputAs[j][i, :,:,:] = self.input_imgs[x:x+512,y:y+512,:]
				outputAs[j][i,:,:,0] = self.target_imgs[j][x:x+512,y:y+512]

		for i in xrange(sizeB):
			#while True:
			x = random.randint(0, self.region_size-image_size-1)
			y = random.randint(0, self.region_size-image_size-1)

			for j in xrange(self.task_num):
				inputBs[j][i, :,:,:] = self.input_imgs[x:x+512,y:y+512,:]
				outputBs[j][i,:,:,0] = self.target_imgs[j][x:x+512,y:y+512]

		return inputAs, outputAs, inputBs, outputBs
		


if __name__ == "__main__":
	output_folder1 = "e5/"
	output_folder2 = "e6/"
	model_folder = "model_v1plus/"


	random.seed(123)
	with tf.Session() as sess:
		model = MAML(sess)
		if len(sys.argv) > 1:
			model.restoreModel(sys.argv[1])


		#dataloader = DataLoader('/data/songtao/metalearning/dataset/boston_task5_small_highres/', 48, 5, preload = True)
		#dataloader2 = DataLoader('/data/songtao/metalearning/dataset/boston_task5_small_highres/', 48, 5, preload = False)

		#dataloader = DataLoader('/data/songtao/metalearning/dataset/boston_task5_small_highres/', 48, 5, preload = True)
		#dataloader2 = DataLoader('/data/songtao/metalearning/dataset/boston_task5_small_highres/', 48, 5, preload = False)

		dataloader = DataLoader('/data/songtao/metalearning/dataset/boston_task5_small/', 16, 5, preload = True)
		dataloader2 = DataLoader('/data/songtao/metalearning/dataset/boston_task5_small/', 16, 5, preload = False)


		#dataloader = DataLoader('/mnt/ramdisk/boston_task5_whole/', 300, 5, preload = False, dist=[0.38229512917303893, 0.078667556313685313, 0.16881041929031509, 0.059084571340968561, 0.31114232388199203])
		#dataloader2 = DataLoader('/mnt/ramdisk/boston_task5_whole/', 300, 5, preload = False, dist=[0.38229512917303893, 0.078667556313685313, 0.16881041929031509, 0.059084571340968561, 0.31114232388199203])

		#testCases = dataloader2.getTestBatchs(5,20)

		testCase = []
		for i in xrange(5):
			#testCase.append(dataloader.getBatch(5,10,update = True))

			testCase.append(dataloader2.getTestBatchs(5,20))






		step = 0

		losses = {}
		loss_curves = {}

		ts = time()

		last_longterm_loss = 10000
		longterm_loss = 0


		task_lr = [1.0]*5

		while True:
			if step % 5 == 0:
				iA,oA, iB, oB, task_id = dataloader.getBatch(20,20,update = True)
			else:
				iA,oA, iB, oB, task_id = dataloader.getBatch(20,20,update = False)
			#print(step)
			


			ret = model.trainModel(iA,oA,iB,oB, task_lr[task_id])

			loss = ret[1]
			loss_curve = ret[2:]

			if task_id in loss_curves:
				loss_curves[task_id][0] = [loss_curves[task_id][0][ind] + loss_curve[ind] for ind in xrange(len(loss_curve))]
				loss_curves[task_id][1] += 1
			else:
				loss_curves[task_id] = [loss_curve,1]

			
			#print(step,loss)
			step = step + 1

			if task_id in losses:
				losses[task_id][0] += loss
				losses[task_id][1] += 1

			else:
				losses[task_id] = [loss,1]

			if step % 10 == 0:
				print(step)
			if step % 200 == 0 or step == 1:

				if step != 1:
					ss = 0
					cc = 0
					s = 0
					for t in xrange(5):
						if t in losses:
							print(t, losses[t][0]/losses[t][1])
							cc += losses[t][1]
							s += losses[t][0]

						if t in loss_curves:
							ppp = [loss_curves[t][0][ind]/loss_curves[t][1] for ind in xrange(len(loss_curves[t][0]))]
							print('curve', t, ppp)

					losses = {}
					loss_curves = {}


					print(step, "total loss", s/cc)

					longterm_loss += s/cc

					if step % 1000 == 0:
						for t in xrange(5):
							task_lr[t] = 1.0

					if step % 200 == 0:
						model.saveModel(model_folder+"/model%d"%step)


				eeid = 0

				if step % 1000 == 0:
					tmpTestCase = testCase
				else:
					tmpTestCase = testCase[:2]

				test_loss = 0
				eeid = 0
				for testCases in tmpTestCase:
					for i in xrange(5):
						

						testInputs = [testCases[0][i],testCases[1][i], testCases[2][i],testCases[3][i],i]
						#print(testInputs)
						result, result_loss, result_debug = model.runModel(testInputs[0],testInputs[1],testInputs[2], testInputs[3])

						test_loss = test_loss + result_loss * dataloader.distribution[i]

						tmp_img = np.zeros((512,512,3), dtype=np.uint8)
						tmp_img2 = np.zeros((512,512), dtype=np.uint8)

						for ind in xrange(len(testInputs[2])):
							tmp_img = (testInputs[2][ind,:,:,:]*255).reshape(512,512,3).astype(np.uint8)
							Image.fromarray(tmp_img).save(output_folder1+'img_%d_%d_sat.png' % (eeid, ind))
							#if ind < 5:
							#	print(np.amin(result[ind,:,:,:]*255), np.amax(result[ind,:,:,:]*255))

							tmp_img2 = (result[ind,:,:,0]*255).reshape(512,512).astype(np.uint8)
							Image.fromarray(tmp_img2).save(output_folder1+'img_%d_%d_task%d_output.png' % (eeid, ind, testInputs[4]))
							#if ind == 0:
							#	print(tmp_img2)

							tmp_img2 = (result[ind,:,:,0]*255).reshape(512,512)
							#tmp_img2 = (result_debug[ind,:,:,0]*255).reshape(512,512)
							tmp_img2[np.where(tmp_img2>127)] = 255
							tmp_img2[np.where(tmp_img2<=127)] = 0

							# if ind == 0:
							# 	print(tmp_img2)


							#if ind < 5:
							#	print(np.amax(tmp_img2),np.amin(tmp_img2))

							tmp_img2 = tmp_img2.astype(np.uint8)

							Image.fromarray(tmp_img2).save(output_folder1+'img_%d_%d_task%d_output_sharp.png' % (eeid, ind, testInputs[4]))

							tmp_img2 = (testInputs[3][ind,:,:,0]*255).reshape(512,512).astype(np.uint8)
							Image.fromarray(tmp_img2).save(output_folder1+'img_%d_%d_task%d_target.png' % (eeid, ind, testInputs[4]))
							
					eeid += 1

				test_loss/=len(tmpTestCase)

				print("test_loss", test_loss)

				if step % 1000 == 0:
					print("longterm_loss",longterm_loss/5," lr is", model.meta_lr_val)
					print("test_loss", test_loss)

					longterm_loss = test_loss  # use test_loss as trigger

					# if longterm_loss > last_longterm_loss*0.99 and step % 10000 == 0:
					# 	#reduce learning rate
					# 	if model.meta_lr_val > 0.00001:
					# 		model.meta_lr_val = model.meta_lr_val / 2
					# 		longterm_loss = 10000 # don't reduce the learning rate two times
					# 	# new_lr = sess.run([model.change_lr_op], feed_dict = {model.meta_lr_val:model.meta_lr_val})

					# 	print("change lr to", model.meta_lr_val)

					last_longterm_loss = longterm_loss
					longterm_loss = 0



				# if step % 1000 == 0:
				# 	tmpTestCase = testCase
				# else:
				# 	tmpTestCase = testCase[0:2]

				# for testInputs in tmpTestCase:
				# 	result,result_loss, result_debug = model.runModel(testInputs[0],testInputs[1],testInputs[2])
				# 	tmp_img = np.zeros((512,512,3), dtype=np.uint8)
				# 	tmp_img2 = np.zeros((512,512), dtype=np.uint8)

				# 	for ind in xrange(len(testInputs[2])):
				# 		tmp_img = (testInputs[2][ind,:,:,:]*255).reshape(512,512,3).astype(np.uint8)
				# 		Image.fromarray(tmp_img).save(output_folder2+'img_%d_%d_sat.png' % (eeid, ind))
				# 		#if ind < 5:
				# 		#	print(np.amin(result[ind,:,:,:]*255), np.amax(result[ind,:,:,:]*255))

				# 		tmp_img2 = (result[ind,:,:,0]*255).reshape(512,512).astype(np.uint8)
				# 		Image.fromarray(tmp_img2).save(output_folder2+'img_%d_%d_task%d_output.png' % (eeid, ind, testInputs[4]))
				# 		#if ind == 0:
				# 		#	print(tmp_img2)

				# 		tmp_img2 = (result[ind,:,:,0]*255).reshape(512,512)
				# 		#tmp_img2 = (result_debug[ind,:,:,0]*255).reshape(512,512)
				# 		tmp_img2[np.where(tmp_img2>127)] = 255
				# 		tmp_img2[np.where(tmp_img2<=127)] = 0

				# 		# if ind == 0:
				# 		# 	print(tmp_img2)


				# 		#if ind < 5:
				# 		#	print(np.amax(tmp_img2),np.amin(tmp_img2))

				# 		tmp_img2 = tmp_img2.astype(np.uint8)

				# 		Image.fromarray(tmp_img2).save(output_folder2+'img_%d_%d_task%d_output_sharp.png' % (eeid, ind, testInputs[4]))

				# 		tmp_img2 = (testInputs[3][ind,:,:,0]*255).reshape(512,512).astype(np.uint8)
				# 		Image.fromarray(tmp_img2).save(output_folder2+'img_%d_%d_task%d_target.png' % (eeid, ind, testInputs[4]))
						

				# 	eeid += 1
				# print("Dump results done")


				t_int = time() - ts 
				ts = time()

				print("iterations per hour", 3600/t_int * 200, t_int, dataloader.total_time, dataloader.total_time/t_int)
				dataloader.total_time = 0

	










