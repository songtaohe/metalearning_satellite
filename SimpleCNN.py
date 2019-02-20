#
#
# This one looks good! It converged!
# 20180401

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
import DataLoader20180429 as MainDataLoader

image_size = 512


global_dist = [0,0.5,0.0,0.5,0]
# Meta Examples Num during training 5 (high quality examples)
# Learning Step 10?


def lrelu(x, alpha):
	return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def sigmoid_weighted(color_intensity, color_central = 74, color_scale = 0.1):
	return tf.nn.sigmoid((color_intensity - color_central)*color_scale) * 5.0 






def L2Loss(a,b):
	return tf.reduce_mean(tf.multiply(tf.square(a-b),b+0.1)) 

def CrossEntropy(a,b):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.concat([b,1-b],3), logits=a))
	#return tf.reduce_mean(tf.multiply(tf.square(a-b),b+0.1)) 



class MAML(object):
	def __init__(self, sess, num_test_updates = 20, inner_lr = 0.001):
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


		# self.inputA = tflearn.input_data(shape = [None, image_size, image_size, 3])
		# self.outputA = tflearn.input_data(shape = [None, image_size, image_size, 1])
		# self.inputB = tflearn.input_data(shape = [None, image_size, image_size, 3])
		# self.outputB = tflearn.input_data(shape = [None, image_size, image_size, 1])

		self.inputA = tflearn.input_data(shape = [None, None, None, 3])
		self.outputA = tflearn.input_data(shape = [None, None, None, 1])
		self.inputB = tflearn.input_data(shape = [None, None, None, 3])
		self.outputB = tflearn.input_data(shape = [None, None, None, 1])



		# with tf.variable_scope("foo", reuse=False):
		# 	self.task_losses, self.task_outputs, _, self.groupA_loss = CNNmodel.buildMetaBlockV1_old(self.inputA, self.outputA, self.inputB, self.outputB, inner_step = self.num_updates, inner_lr = self.inner_lr)
		# with tf.variable_scope("foo", reuse=True):
		# 	self.task_losses_test, self.task_test_outputs, self.debug_inner_output, self.groupA_loss_test = CNNmodel.buildMetaBlockV1_old(self.inputA, self.outputA, self.inputB, self.outputB, inner_step = self.num_test_updates, inner_lr = self.inner_lr_test, layer_st = 10, layer_ed = 13)


		with tf.variable_scope("foo", reuse=False):
			self.baseline_output_, _ = CNNmodel.build_simple_net_512_4_V1(self.inputA, prefix="first_", deconv=True)

		self.baseline_output = tf.nn.softmax(self.baseline_output_)
		self.baseline_loss = CrossEntropy(self.baseline_output_, self.outputA)
		self.baseline_train_op = tf.train.AdamOptimizer(learning_rate=self.meta_lr).minimize(self.baseline_loss)



		# optimizer = tf.train.AdamOptimizer(self.meta_lr)
		# self.gvs = gvs = optimizer.compute_gradients(self.task_losses[self.num_updates-1])
		# self.metatrain_op = optimizer.apply_gradients(gvs)


		

		### self.metatrain_op = tf.train.AdamOptimizer(learning_rate=self.meta_lr).minimize(self.task_losses[self.num_updates-1])




		#self.metatrain_groupA_op = tf.train.AdamOptimizer(learning_rate=self.meta_lr).minimize(self.groupA_loss)



		#self.metatrain_op = tf.train.AdamOptimizer(learning_rate=self.meta_lr).minimize(self.task_losses[self.num_updates-1] + (self.task_losses[self.num_updates-1] - self.task_losses[self.num_updates-2]))


		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep=100)





		self.summary_loss = []
		self.test_loss =  tf.placeholder(tf.float32)
		self.train_loss =  tf.placeholder(tf.float32)
		self.lr =  tf.placeholder(tf.float32)

		self.summary_loss.append(tf.summary.scalar('test_loss', self.test_loss))
		self.summary_loss.append(tf.summary.scalar('train_loss', self.train_loss))
		self.summary_loss.append(tf.summary.scalar('lr', self.lr))

		self.merged_summary = tf.summary.merge_all()


		# self.change_lr_op = self.meta_lr.assign(self.meta_lr_val)

		# self.sess.run([self.change_lr_op], feed_dict = {self.meta_lr_val:self.meta_lr_val})


	def trainModel(self, inputA, outputA, inputB, outputB, scale = 1.0):

		return self.sess.run([self.metatrain_op, self.task_losses[self.num_updates-1]] + self.task_losses, feed_dict = {self.inputA:inputA, self.outputA:outputA, self.inputB:inputB, self.outputB:outputB, self.meta_lr:self.meta_lr_val * scale})

	def trainModelGroupA(self, inputA, outputA, scale = 1.0):

		return self.sess.run([self.metatrain_groupA_op, self.groupA_loss], feed_dict = {self.inputA:inputA, self.outputA:outputA, self.meta_lr:self.meta_lr_val * scale})

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


	def addLog(self, test_loss, train_loss, lr):
		return self.sess.run(self.merged_summary , feed_dict = {self.test_loss:test_loss, self.train_loss: train_loss, self.lr:lr})

	def saveModel(self, path):
		self.saver.save(self.sess, path)

	def restoreModel(self, path):
		self.saver.restore(self.sess, path)


if __name__ == "__main__":

	name = "ModelV1_20180504_multidataset_osm_deepglobal_dstl_incremental_converge"
	run = name+"_run1"

	Popen("mkdir -p %s" % name, shell=True).wait()

	output_folder1 = name+"/e1/"
	output_folder2 = name+"/e2/"
	model_folder = name+"/model/"
	#log_folder = name+"/log"
	log_folder = "alllogs/log"

	Popen("mkdir -p %s" % output_folder1, shell=True).wait()
	Popen("mkdir -p %s" % output_folder2, shell=True).wait()
	Popen("mkdir -p %s" % model_folder, shell=True).wait()
	Popen("mkdir -p %s" % log_folder, shell=True).wait()


	random.seed(321)

	with tf.Session() as sess:
		model = MAML(sess)
		
		if len(sys.argv) > 1:
			model.restoreModel(sys.argv[1])

		writer = tf.summary.FileWriter(log_folder+"/"+run, sess.graph)
		Popen("pkill tensorboard", shell=True).wait()
		sleep(1)
		print("tensorboard --logdir=%s"%log_folder)
		logserver = Popen("tensorboard --logdir=%s"%log_folder, shell=True)


		# path = "/mnt/satellite/global/"
		# dataloader1 = DataLoader(path+'boston/', 36, 5, preload = False)
		# dataloader2 = DataLoader(path+'la/', 36, 5, preload = False)
		# dataloader3 = DataLoader(path+'chicago/', 36, 5, preload = False)
		# #dataloader4 = DataLoader('/data/songtao/metalearning/dataset/global/dc/', 36, 5, preload = False)
		# #dataloader5 = DataLoader('/data/songtao/metalearning/dataset/global/houston/', 36, 5, preload = False)

		# dataloader = DataLoaderWrapper([dataloader1,dataloader2,dataloader3])


		OSM_folder = ['/data/songtao/metalearning/dataset/global/boston/','/data/songtao/metalearning/dataset/global/chicago/','/data/songtao/metalearning/dataset/global/la/']

		loaderOSMroad = MainDataLoader.LoaderOSM(OSM_folder, subtask=2)
		loaderOSMBuilding = MainDataLoader.LoaderOSM(OSM_folder, subtask=4)

		loaderDeepGlobalLand0 = MainDataLoader.LoaderCVPRContestLandCover('/data/songtao/metalearning/cvprcontextDataset/landcover/land-train/', subtask=0)
		loaderDeepGlobalLand1 = MainDataLoader.LoaderCVPRContestLandCover('/data/songtao/metalearning/cvprcontextDataset/landcover/land-train/', subtask=1)
		loaderDeepGlobalLand2 = MainDataLoader.LoaderCVPRContestLandCover('/data/songtao/metalearning/cvprcontextDataset/landcover/land-train/', subtask=2)
		loaderDeepGlobalLand3 = MainDataLoader.LoaderCVPRContestLandCover('/data/songtao/metalearning/cvprcontextDataset/landcover/land-train/', subtask=3)
		loaderDeepGlobalLand4 = MainDataLoader.LoaderCVPRContestLandCover('/data/songtao/metalearning/cvprcontextDataset/landcover/land-train/', subtask=4)
		loaderDeepGlobalLand5 = MainDataLoader.LoaderCVPRContestLandCover('/data/songtao/metalearning/cvprcontextDataset/landcover/land-train/', subtask=5)

		loaderDeepGlobalRoadDetection = MainDataLoader.LoaderCVPRContestRoadDetection('/data/songtao/metalearning/cvprcontextDataset/road/train/')

		loaderDSTLs = [MainDataLoader.LoaderKaggleDSTL('/data/songtao/metalearning/public_dataset/dstl_ready_to_use/', subtask=i) for i in xrange(1,11)]

		loaders = [loaderOSMroad,loaderOSMBuilding,loaderDeepGlobalLand0,loaderDeepGlobalLand1,loaderDeepGlobalLand2,loaderDeepGlobalLand3,loaderDeepGlobalLand4,loaderDeepGlobalLand5,loaderDeepGlobalRoadDetection] + loaderDSTLs

		dataloader = MainDataLoader.DataLoaderMultiplyTask(loaders)


		#preload_nums = [100,100, 100,100,100,100,100,100, 100]

		#task_p = [0.15,0.15, 0.1,0.1,0.1,0.1,0.1,0.1, 0.1]
		task_p = [1.0/len(loaders) for x in xrange(len(loaders))]

		print("Preload")
		t0 = time()
		dataloader.preload(100)
		print("Preload Done", time()-t0)

		t0 = time()
		testCase = [dataloader.loadBatchFromTask(i,5,10) for i in xrange(len(loaders))]
		testCase += [dataloader.loadBatchFromTask(i,5,10) for i in xrange(len(loaders))]
		print("Sample Test Data Done", time()-t0)



		step = 0

		losses = {}
		loss_curves = {}

		ts = time()

		last_longterm_loss = 10000
		longterm_loss = 0
		test_loss = 0
		while True:
			
			if step % 500 == 0 and step != 0:
				t0 = time()
				dataloader.preload(200)
				print("Step ", step, "Preload Done", time()-t0)

			iA,oA, iB, oB, task_id = dataloader.loadBatch(5,10,p=task_p)
			#print(step)

			ret = model.trainModel(iA,oA,iB,oB, 1.0)

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
				train_loss = 0
				if step != 1:
					ss = 0
					cc = 0
					s = 0
					for t in xrange(len(loaders)):
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
					train_loss = s/cc
					longterm_loss += s/cc

					if step % 400 == 0:
						model.saveModel(model_folder+"/model%d"%step)

				eeid = 0
	
				#test_num = 2

				# if step % 1000 == 0:
				# 	tmpTestCase = testCase
				# else:
				# 	tmpTestCase = testCase[:test_num]


				if step % 400 == 0 or step < 400:
					test_loss = 0
					#test_loss_first_two = 0

					eeid = 0
					for testCases in testCase:

						result, result_loss, _ = model.runModel(testCases[0],testCases[1],testCases[2], testCases[3])

						tmp_img = np.zeros((512,512,3), dtype=np.uint8)
						tmp_img2 = np.zeros((512,512), dtype=np.uint8)

						# output input
						for ind in xrange(len(testCases[0])):
							tmp_img = (testCases[0][ind,:,:,:]*255).astype(np.uint8)
							Image.fromarray(tmp_img).save(output_folder1+'img_%d_%d_sat.png' % (eeid, ind))

							tmp_img2 = (testCases[1][ind,:,:,0]*255).astype(np.uint8)
							Image.fromarray(tmp_img2).save(output_folder1+'img_%d_%d_target.png' % (eeid, ind))

						for ind in xrange(len(testCases[2])):
							tmp_img = (testCases[2][ind,:,:,:]*255).astype(np.uint8)
							Image.fromarray(tmp_img).save(output_folder1+'img_%d_%d_sat.png' % (eeid, ind+len(testCases[0])))

							tmp_img2 = (testCases[3][ind,:,:,0]*255).astype(np.uint8)
							Image.fromarray(tmp_img2).save(output_folder1+'img_%d_%d_target.png' % (eeid, ind+len(testCases[0])))

							tmp_img2 = (result[ind,:,:,0]*255).astype(np.uint8)
							Image.fromarray(tmp_img2).save(output_folder1+'img_%d_%d_output.png' % (eeid, ind+len(testCases[0])))

						test_loss = test_loss + result_loss
						eeid += 1

					test_loss/=eeid
				

				print("test_loss", test_loss)

				if step>1:
					summary = model.addLog(test_loss, train_loss, model.meta_lr_val)
					writer.add_summary(summary, step)
				

				if step % 5000 == 0:



					print("longterm_loss",longterm_loss/5," lr is", model.meta_lr_val)

					longterm_loss = test_loss

					print(longterm_loss, last_longterm_loss)

					if longterm_loss > last_longterm_loss*1.0:
						#reduce learning rate
						#if model.meta_lr_val > 0.0001:
						if model.meta_lr_val > 0.000005:
							#model.meta_lr_val = model.meta_lr_val / 2
							longterm_loss = 10000 # don't reduce the learning rate two times
						# new_lr = sess.run([model.change_lr_op], feed_dict = {model.meta_lr_val:model.meta_lr_val})

						print("change lr to", model.meta_lr_val)

					last_longterm_loss = longterm_loss
					longterm_loss = 0

					if step % 5000 == 0:
						model.meta_lr_val = model.meta_lr_val / 1.1
						print("change lr to", model.meta_lr_val)






				t_int = time() - ts 
				ts = time()
				print("iterations per hour", 3600/t_int * 200)
				#print("iterations per hour", 3600/t_int * 200, t_int, dataloader.total_time, dataloader.total_time/t_int)
				#dataloader.total_time = 0

	




		writer.close()





