import os, datetime
import numpy as np
import tensorflow as tf
import tflearn
from tensorflow.contrib.layers.python.layers import batch_norm
from PIL import Image
import tflearn
import random
import scipy
from time import time,sleep
import sys
import CNNmodel
from subprocess import Popen
import inputFilter 

image_size = 512

def lrelu(x, alpha):
	return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def sigmoid_weighted(color_intensity, color_central = 74, color_scale = 0.1):
	return tf.nn.sigmoid((color_intensity - color_central)*color_scale) * 5.0 


def L2Loss(a,b):
	return tf.reduce_mean(tf.multiply(tf.square(a-b),b+0.1)) 

def CrossEntropy(a,b):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.concat([b,1-b],3), logits=a))
	#return tf.reduce_mean(tf.multiply(tf.square(a-b),b+0.1)) 



class MAMLBase(object):
	def __init__(self, sess, num_test_updates = 20, inner_lr = 0.001, meta_block = CNNmodel.buildMetaBlockV1, cnn_model = CNNmodel.build_unet512_12_V1, build_cnn_model = CNNmodel.build_unet512_12_V1_fixed, run_cnn_model = CNNmodel.run_unet512_12_V1_fixed, reuse=False):
		#self.num_updates = 10
		self.num_updates = 10
		self.num_test_updates = 20
		self.num_test_updates = num_test_updates
		self.meta_lr_val = 0.001
		self.meta_lr_val = 0.00002

		self.meta_lr_val = 0.0001/2

		self.meta_lr = tf.placeholder(tf.float32, shape=[])

		self.inner_lr = self.meta_lr * 10 
		self.inner_lr = 0.001
		self.inner_lr_test = 0.001

		self.sess = sess

		self.inputA = tflearn.input_data(shape = [None, image_size, image_size, 3])
		self.outputA = tflearn.input_data(shape = [None, image_size, image_size, 1])
		self.inputB = tflearn.input_data(shape = [None, image_size, image_size, 3])
		self.outputB = tflearn.input_data(shape = [None, image_size, image_size, 1])

		with tf.variable_scope("foo", reuse=reuse):
			self.task_losses, self.task_outputs, _, self.groupA_loss,self.groupA_losses,self.max_grad = meta_block(self.inputA, self.outputA, self.inputB, self.outputB, inner_step = self.num_updates, inner_lr = self.inner_lr, build_cnn_model = build_cnn_model, run_cnn_model = run_cnn_model)
		with tf.variable_scope("foo", reuse=True):
			self.task_losses_test, self.task_test_outputs, self.debug_inner_output, self.groupA_loss_test,_,_ = meta_block(self.inputA, self.outputA, self.inputB, self.outputB, inner_step = self.num_test_updates, inner_lr = self.inner_lr_test, layer_st = 10, layer_ed = 13, build_cnn_model = build_cnn_model, run_cnn_model = run_cnn_model)


		with tf.variable_scope("foo", reuse=True):
			self.baseline_output_, _ = cnn_model(self.inputA, prefix="first_", deconv=True)

		self.baseline_output = tf.nn.softmax(self.baseline_output_)
		self.baseline_loss = CrossEntropy(self.baseline_output_, self.outputA)
		self.baseline_train_op = tf.train.AdamOptimizer(learning_rate=self.meta_lr).minimize(self.baseline_loss)



		# optimizer = tf.train.AdamOptimizer(self.meta_lr)
		# self.gvs = gvs = optimizer.compute_gradients(self.task_losses[self.num_updates-1])
		# self.metatrain_op = optimizer.apply_gradients(gvs)

		self.metatrain_op = tf.train.AdamOptimizer(learning_rate=self.meta_lr).minimize(self.task_losses[self.num_updates-1])

		# self.opt = tf.train.AdamOptimizer(learning_rate=self.meta_lr)
		# grads = self.opt.compute_gradients(self.groupA_loss)
		# self.train_max_grad = 0
		# for grad in grads:
		# 	self.train_max_grad = tf.maximum(tf.reduce_max(tf.abs(grad)), self.train_max_grad)

		# self.metatrain_groupA_op = self.opt.apply_gradients(grads) # normal sgd training 



		# #self.metatrain_groupA_op = tf.train.AdamOptimizer(learning_rate=self.meta_lr).minimize(self.groupA_loss)



		#self.metatrain_op = tf.train.AdamOptimizer(learning_rate=self.meta_lr).minimize(self.task_losses[self.num_updates-1] + (self.task_losses[self.num_updates-1] - self.task_losses[self.num_updates-2]))


		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep=100)





		self.summary_loss = []
		self.test_loss =  tf.placeholder(tf.float32)
		self.train_loss =  tf.placeholder(tf.float32)
		self.lr =  tf.placeholder(tf.float32)
		self.max_g1 =  tf.placeholder(tf.float32)
		self.max_g2 =  tf.placeholder(tf.float32)

		self.summary_loss.append(tf.summary.scalar('loss/test', self.test_loss))
		self.summary_loss.append(tf.summary.scalar('loss/train', self.train_loss))
		self.summary_loss.append(tf.summary.scalar('lr', self.lr))
		self.summary_loss.append(tf.summary.scalar('grad/train', self.max_g1))
		self.summary_loss.append(tf.summary.scalar('grad/maml', self.max_g2))

		self.merged_summary = tf.summary.merge_all()


		# self.change_lr_op = self.meta_lr.assign(self.meta_lr_val)

		# self.sess.run([self.change_lr_op], feed_dict = {self.meta_lr_val:self.meta_lr_val})


	def trainModel(self, inputA, outputA, inputB, outputB, scale = 1.0):

		return self.sess.run([self.metatrain_op, self.task_losses[self.num_updates-1]] + self.task_losses, feed_dict = {self.inputA:inputA, self.outputA:outputA, self.inputB:inputB, self.outputB:outputB, self.meta_lr:self.meta_lr_val * scale})

	def trainModelGroupA(self, inputA, outputA, scale = 1.0):

		return self.sess.run([self.train_max_grad,self.max_grad]+[self.metatrain_groupA_op, self.groupA_loss] + self.groupA_losses, feed_dict = {self.inputA:inputA, self.outputA:outputA, self.meta_lr:self.meta_lr_val * scale})

	# def runModel(self, inputA, outputA, inputB):
	# 	return self.sess.run([self.task_test_outputs[self.num_test_updates-1], self.debug_inner_output], feed_dict = {self.inputA:inputA, self.outputA:outputA, self.inputB:inputB,self.meta_lr:self.meta_lr_val})

	def runModel(self, inputA, outputA, inputB, outputB):
		return self.sess.run([self.task_test_outputs[self.num_test_updates-1], self.task_losses_test[self.num_test_updates-1], self.debug_inner_output], feed_dict = {self.inputA:inputA, self.outputA:outputA, self.inputB:inputB,self.outputB:outputB, self.meta_lr:self.meta_lr_val})

	def trainBaselineModel(self, inputA, outputA, scale = 1.0):

		return self.sess.run([self.baseline_train_op, self.baseline_loss], feed_dict = {self.inputA:inputA, self.outputA:outputA, self.meta_lr:self.meta_lr_val * scale})

	# def runModel(self, inputA, outputA, inputB):
	# 	return self.sess.run([self.task_test_outputs[self.num_test_updates-1], self.debug_inner_output], feed_dict = {self.inputA:inputA, self.outputA:outputA, self.inputB:inputB,self.meta_lr:self.meta_lr_val})

	def runBaselineModel(self, inputA, outputA):
		return self.sess.run([self.baseline_output, self.baseline_loss], feed_dict = {self.inputA:inputA, self.outputA:outputA, self.meta_lr:self.meta_lr_val})


	def addLog(self, test_loss, train_loss, lr,g1=0,g2=0):
		return self.sess.run(self.merged_summary , feed_dict = {self.test_loss:test_loss, self.train_loss: train_loss, self.lr:lr, self.max_g1:g1, self.max_g2:g2})

	def saveModel(self, path):
		self.saver.save(self.sess, path)

	def restoreModel(self, path):
		self.saver.restore(self.sess, path)


class MAMLFirstOrder(MAMLBase):
	def __init__(self, sess, num_test_updates = 20, inner_lr = 0.001):
		super(MAMLFirstOrder, self).__init__(sess, num_test_updates, inner_lr, meta_block = CNNmodel.buildMetaBlockV1, cnn_model=CNNmodel.build_unet512_12_V1, build_cnn_model = CNNmodel.build_unet512_12_V1_fixed, run_cnn_model = CNNmodel.run_unet512_12_V1_fixed)


class MAMLFirstOrderMultiResolution(MAMLBase):
	def __init__(self, sess, num_test_updates = 20, inner_lr = 0.001):
		super(MAMLFirstOrderMultiResolution, self).__init__(sess, num_test_updates, inner_lr, CNNmodel.buildMetaBlockV1, CNNmodel.build_unet512_12_V1, build_cnn_model = CNNmodel.build_unet512_12_V1_fixed, run_cnn_model = CNNmodel.run_unet512_12_V1_fixed)




# class Reptile(MAMLBase):
# 	def __init__(self, sess, num_test_updates = 20, inner_lr = 0.001):
# 		super().__init__(num_test_updates, inner_lr, CNNmodel.buildMetaBlockV1, CNNmodel.build_unet512_12_V1)





