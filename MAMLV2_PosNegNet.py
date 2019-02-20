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

image_size = 512


global_dist = [0,0.5,0.0,0.5,0]
# Meta Examples Num during training 5 (high quality examples)
# Learning Step 10?




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

		self.meta_lr_val = 0.0001
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


		self.inputA_IntraClass = tflearn.input_data(shape = [None, image_size, image_size, 4])
		self.outputA_IntraClass = tflearn.input_data(shape = [None, image_size, image_size, 1])
		self.inputB_IntraClass = tflearn.input_data(shape = [None, image_size, image_size, 4])
		self.outputB_IntraClass = tflearn.input_data(shape = [None, image_size, image_size, 1])



		

		with tf.variable_scope("foo", reuse=False):
			self.task_losses, self.task_outputs, _ = CNNmodel.buildMetaBlockV1(self.inputA, self.outputA, self.inputB, self.outputB, inner_step = self.num_updates, inner_lr = self.inner_lr)
		with tf.variable_scope("foo", reuse=True):
			self.task_losses_test, self.task_test_outputs, self.debug_inner_output = CNNmodel.buildMetaBlockV1(self.inputA, self.outputA, self.inputB, self.outputB, inner_step = self.num_test_updates, inner_lr = self.inner_lr_test, layer_st = 10, layer_ed = 13)


		with tf.variable_scope("foo", reuse=True):
			self.baseline_output_, _ = CNNmodel.build_unet512_12_V1_fixed(self.inputA, prefix="first_", deconv=True)

		self.baseline_output = tf.nn.softmax(self.baseline_output_)
		self.baseline_loss = CrossEntropy(self.baseline_output_, self.outputA)
		self.baseline_train_op = tf.train.AdamOptimizer(learning_rate=self.meta_lr).minimize(self.baseline_loss)



		# optimizer = tf.train.AdamOptimizer(self.meta_lr)
		# self.gvs = gvs = optimizer.compute_gradients(self.task_losses[self.num_updates-1])
		# self.metatrain_op = optimizer.apply_gradients(gvs)


		self.metatrain_op = tf.train.AdamOptimizer(learning_rate=self.meta_lr).minimize(self.task_losses[self.num_updates-1])
		#self.metatrain_op = tf.train.AdamOptimizer(learning_rate=self.meta_lr).minimize(self.task_losses[self.num_updates-1] + (self.task_losses[self.num_updates-1] - self.task_losses[self.num_updates-2]))


		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep=100)



		# IntraClass Classification 

		self.num_updates_IntraClass = 5
		self.num_test_updates_IntraClass = 20

		with tf.variable_scope("PosNeg", reuse=False):
			self.IntraClass_losses, self.IntraClass_outputs, _ = CNNmodel.buildMetaBlockV1(self.inputA_IntraClass, self.outputA_IntraClass, self.inputB_IntraClass, self.outputB_IntraClass, inner_step = self.num_updates_IntraClass, inner_lr = self.inner_lr, layer_st = 5, layer_ed=8, build_cnnmodel=CNNmodel.build_unet512_IntraClass_8, run_cnnmodel=CNNmodel.run_unet512_IntraClass_8, inputdim=4)
		
		with tf.variable_scope("PosNeg", reuse=True):
			self.IntraClass_losses_test, self.IntraClass_outputs_test, _ = CNNmodel.buildMetaBlockV1(self.inputA_IntraClass, self.outputA_IntraClass, self.inputB_IntraClass, self.outputB_IntraClass, inner_step = self.num_test_updates_IntraClass, inner_lr = self.inner_lr, layer_st = 5, layer_ed=8, build_cnnmodel=CNNmodel.build_unet512_IntraClass_8, run_cnnmodel=CNNmodel.run_unet512_IntraClass_8,inputdim=4)
		
		self.metatrain_IntraClass_op = tf.train.AdamOptimizer(learning_rate=self.meta_lr).minimize(self.IntraClass_losses[self.num_updates_IntraClass-1])
		


		self.sess.run(tf.global_variables_initializer())
		self.saverAll = tf.train.Saver(max_to_keep=100)


		self.summary_loss = []
		self.test_loss =  tf.placeholder(tf.float32)
		self.train_loss =  tf.placeholder(tf.float32)
		self.lr =  tf.placeholder(tf.float32)
		self.PosnegLoss = tf.placeholder(tf.float32)
		self.IntraClass_loss = tf.placeholder(tf.float32)

		self.summary_loss.append(tf.summary.scalar('test_loss', self.test_loss))
		self.summary_loss.append(tf.summary.scalar('train_loss', self.train_loss))
		self.summary_loss.append(tf.summary.scalar('lr', self.lr))
		self.summary_loss.append(tf.summary.scalar('PosNeg', self.PosnegLoss))
		self.summary_loss.append(tf.summary.scalar('IntraClass_loss', self.IntraClass_loss))

		self.merged_summary = tf.summary.merge_all()


		# self.change_lr_op = self.meta_lr.assign(self.meta_lr_val)

		# self.sess.run([self.change_lr_op], feed_dict = {self.meta_lr_val:self.meta_lr_val})


	def trainModel(self, inputA, outputA, inputB, outputB, scale = 1.0):

		return self.sess.run([self.metatrain_op, self.task_losses[self.num_updates-1]] + self.task_losses, feed_dict = {self.inputA:inputA, self.outputA:outputA, self.inputB:inputB, self.outputB:outputB, self.meta_lr:self.meta_lr_val * scale})

	def runModel(self, inputA, outputA, inputB, outputB):
		return self.sess.run([self.task_test_outputs[self.num_test_updates-1], self.task_losses_test[self.num_test_updates-1], self.debug_inner_output], feed_dict = {self.inputA:inputA, self.outputA:outputA, self.inputB:inputB,self.outputB:outputB, self.meta_lr:self.meta_lr_val})

	def trainModelIntraClass(self, inputA, outputA, inputB, outputB, scale = 1.0):

		return self.sess.run([self.metatrain_IntraClass_op, self.IntraClass_losses[self.num_updates_IntraClass-1]] + self.IntraClass_losses, feed_dict = {self.inputA_IntraClass:inputA, self.outputA_IntraClass:outputA, self.inputB_IntraClass:inputB, self.outputB_IntraClass:outputB, self.meta_lr:self.meta_lr_val * scale})

	def runModelIntraClass(self, inputA, outputA, inputB, outputB):
		return self.sess.run([self.IntraClass_outputs_test[self.num_test_updates_IntraClass-1], self.IntraClass_losses_test[self.num_test_updates_IntraClass-1]], feed_dict = {self.inputA_IntraClass:inputA, self.outputA_IntraClass:outputA, self.inputB_IntraClass:inputB,self.outputB_IntraClass:outputB, self.meta_lr:self.meta_lr_val})




	def trainBaselineModel(self, inputA, outputA, scale = 1.0):

		return self.sess.run([self.baseline_train_op, self.baseline_loss], feed_dict = {self.inputA:inputA, self.outputA:outputA, self.meta_lr:self.meta_lr_val * scale})

	# def runModel(self, inputA, outputA, inputB):
	# 	return self.sess.run([self.task_test_outputs[self.num_test_updates-1], self.debug_inner_output], feed_dict = {self.inputA:inputA, self.outputA:outputA, self.inputB:inputB,self.meta_lr:self.meta_lr_val})

	def runBaselineModel(self, inputA, outputA):
		return self.sess.run([self.baseline_output, self.baseline_loss], feed_dict = {self.inputA:inputA, self.outputA:outputA, self.meta_lr:self.meta_lr_val})


	def addLog(self, test_loss, train_loss, lr, PosnegLoss,IntraClass_loss):
		return self.sess.run(self.merged_summary , feed_dict = {self.test_loss:test_loss, self.train_loss: train_loss, self.lr:lr, self.PosnegLoss:PosnegLoss, self.IntraClass_loss:IntraClass_loss})

	def saveModel(self, path):
		self.saver.save(self.sess, path)

	def restoreModel(self, path):
		self.saver.restore(self.sess, path)

	def saveModelAll(self, path):
		self.saverAll.save(self.sess, path)

	def restoreModelAll(self, path):
		self.saverAll.restore(self.sess, path)


class DataLoaderWrapper(object):
	def __init__(self, dataloaders):
		self.dataloaders = dataloaders 
		self.distribution = global_dist
		self.total_time = 0
	def getInstance(self):
		return np.random.choice(self.dataloaders)

	def getBatch(self, sizeA=20, sizeB=20, update=True, allTask = False):
		ts = time()
		ret = self.getInstance().getBatch(sizeA, sizeB, update, allTask)
		self.total_time += time() - ts
		return ret 

	def getTestBatchs(self, sizeA=20, sizeB=20):
		ts = time()
		ret = self.getInstance().getTestBatchs(sizeA,sizeB)
		self.total_time += time() - ts

		return ret


class DataLoader(object):
	def __init__(self, foldername, region_num, task_num, preload = False, dist = [0.2,0.2,0.2,0.2,0.2]):
		self.region_num = region_num 
		self.task_num = task_num
		self.region_size = 4096
		self.foldername = foldername
		self.preload = preload
		self.cc = 0
		self.interval = 20
		self.total_time = 0


		self.bad_regions = []
		self.distribution = [0.0] * task_num
		self.distribution = global_dist

		if preload == False:
			self.distribution = global_dist
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

		self.distribution[0] = 0
		self.distribution[2] = 0
		self.distribution[4] = 0

		for i in xrange(task_num):
			d_sum += self.distribution[i]

		for i in xrange(task_num):
			self.distribution[i] /= d_sum 

		print("Train Distribution", self.distribution)


		self.bad_regions = []
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
			region_id = random.randint(0, self.region_num - 1)

			retry_c = 0

			# Sample A
			for i in xrange(sizeA):
				while True:
					
					x = random.randint(0, self.region_size-image_size-1)
					y = random.randint(0, self.region_size-image_size-1)

					v_sum = np.sum(self.target_imgs[region_id][task_id][x:x+512,y:y+512])
					if v_sum > 512*512*0.05*0.05:
						break

					retry_c += 1
					if retry_c > 100:
						break


				inputA[i, :,:,:] = self.input_imgs[region_id][x:x+512,y:y+512,:]
				outputA[i,:,:,0] = self.target_imgs[region_id][task_id][x:x+512,y:y+512]

				#	break
			retry_c = 0
			# Sample B
			for i in xrange(sizeB):
				while True:
					#region_id = random.randint(0, self.region_num - 1)
					x = random.randint(0, self.region_size-image_size-1)
					y = random.randint(0, self.region_size-image_size-1)

					v_sum = np.sum(self.target_imgs[region_id][task_id][x:x+512,y:y+512])
					if v_sum > 512*512*0.05*0.05:
						break

					retry_c += 1
					if retry_c > 100:
						break

				inputB[i, :,:,:] = self.input_imgs[region_id][x:x+512,y:y+512,:]
				outputB[i,:,:,0] = self.target_imgs[region_id][task_id][x:x+512,y:y+512]

				#break
		else:
			# Sample A
			while True:
				#if update == True:
				region_id = 0
				if self.cc % 5 == 0 or self.cc == 1 or update == True:
					foldername = self.foldername
					while True:
						region_id = random.randint(0, self.region_num - 1)
						if region_id in self.bad_regions:
							continue 
						break
					while True:
						style_id = random.randint(0,9)
						try:
							self.input_imgs = scipy.ndimage.imread(foldername + "/"+"region%d_sat%d.png"%(region_id,style_id)).astype(np.float)/255.0
						except:
							continue 

						break

					#style_id = random.randint(0,9)
					#self.input_imgs = scipy.ndimage.imread(foldername + "/"+"region%d_sat%d.png"%(region_id,style_id)).astype(np.float)/255.0
		
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
						if retry_c > 30:
							self.bad_regions.append(region_id)
							retry_c = -1
							break
						if i >= sizeA*0.6:
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
				for i in xrange(sizeB):
					retry_c = 0
					while True:
						x = random.randint(0, self.region_size-image_size-1)
						y = random.randint(0, self.region_size-image_size-1)
						v_sum = np.sum(self.target_imgs[task_id][x:x+512,y:y+512])
						if v_sum > 512*512*0.05*0.05:
							break
						retry_c = retry_c + 1
						if retry_c > 30:
							self.bad_regions.append(region_id)
							retry_c = -1
							break
						if i >= sizeB*0.6:
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

		while True:
			style_id = random.randint(0,9)
			print(style_id)
			try:
				self.input_imgs = scipy.ndimage.imread(foldername + "/"+"region%d_sat%d.png"%(region_id,style_id)).astype(np.float)/255.0
			except:
				continue 

			break
		#self.input_imgs = inputFilter.applyFilter(self.input_imgs, random.randint(0,7)).astype(np.float)/255.0

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
		


def TestBatch2PosNeg(testBatch, sizeA=5):
	inputAs = []
	inputBs = []
	outputAs = []
	outputBs = []

	for i in xrange(5):
		inputAs.append(np.zeros((sizeA, image_size, image_size, 3)))
		outputAs.append(np.zeros((sizeA, image_size, image_size, 1)))

		inputBs.append(np.zeros((sizeA, image_size, image_size, 3)))
		outputBs.append(np.zeros((sizeA, image_size, image_size, 1)))


	for i in xrange(sizeA):
		for j in xrange(5):
			inputAs[j][i,:,:,:] = testBatch[0][j][i,:,:,:]

			if i == 2:
				outputAs[j][i,:,:,:] = testBatch[1][j][i,:,:,:]

			inputBs[j][i,:,:,:] = testBatch[0][j][i,:,:,:]
			if i == 2:
				outputBs[j][i,:,:,:] = testBatch[1][j][i,:,:,:]

	return inputAs, outputAs, inputBs, outputBs





if __name__ == "__main__":
	Config = "_only_interclass"

	only_interclass = True



	name = "Model_Interclass_20180427_boston_la_chicago_inputfilter"+Config
	run_name = name+"run2"

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
			#model.restoreModel(sys.argv[1])
			model.restoreModelAll(sys.argv[1])

		writer = tf.summary.FileWriter(log_folder+"/"+run_name, sess.graph)

		Popen("pkill tensorboard", shell=True).wait()
		sleep(1)
		print("tensorboard --logdir=%s"%log_folder)
		logserver = Popen("tensorboard --logdir=%s"%log_folder, shell=True)



		#dataloader = DataLoader('/data/songtao/metalearning/dataset/boston_task5_small_highres/', 48, 5, preload = True)
		#dataloader2 = DataLoader('/data/songtao/metalearning/dataset/boston_task5_small_highres/', 48, 5, preload = False)

		#dataloader = DataLoader('/data/songtao/metalearning/dataset/boston_task5_small_highres/', 48, 5, preload = True)
		#dataloader2 = DataLoader('/data/songtao/metalearning/dataset/boston_task5_small_highres/', 48, 5, preload = False)

		# dataloader = DataLoader('/data/songtao/metalearning/dataset/global/boston/', 36, 5, preload = True)
		# dataloader2 = DataLoader('/data/songtao/metalearning/dataset/global/boston/', 36, 5, preload = False)


		#dataloader = DataLoader('/mnt/ramdisk/boston_task5_whole/', 300, 5, preload = False, dist=[0.38229512917303893, 0.078667556313685313, 0.16881041929031509, 0.059084571340968561, 0.31114232388199203])
		#dataloader2 = DataLoader('/mnt/ramdisk/boston_task5_whole/', 300, 5, preload = False, dist=[0.38229512917303893, 0.078667556313685313, 0.16881041929031509, 0.059084571340968561, 0.31114232388199203])



		path = "/mnt/satellite/global/"
		path = "/data/songtao/metalearning/dataset/global/"
		dataloader1 = DataLoader(path+'boston/', 36, 5, preload = False)
		dataloader2 = DataLoader(path+'la/', 36, 5, preload = False)
		dataloader3 = DataLoader(path+'chicago/', 36, 5, preload = False)
		#dataloader4 = DataLoader('/data/songtao/metalearning/dataset/global/dc/', 36, 5, preload = False)
		#dataloader5 = DataLoader('/data/songtao/metalearning/dataset/global/houston/', 36, 5, preload = False)

		dataloader = DataLoaderWrapper([dataloader1,dataloader2,dataloader3])


		# testCases = dataloader2.getTestBatchs(5,20)

		# testCase = []
		# for i in xrange(2):
		# 	testCase.append(dataloader.getBatch(5,10,update = True))


		testCase = []
		testCasePosNeg = []
		for i in xrange(10):
			#testCase.append(dataloader.getBatch(5,10,update = True))
			testCase.append(dataloader.getTestBatchs(5,10))
			testCasePosNeg.append(TestBatch2PosNeg(testCase[-1]))

		



		step = 0

		losses = {}
		loss_curves = {}

		ts = time()

		last_longterm_loss = 10000
		longterm_loss = 0

		cool_down = 60000

		task_lr = [1.0]*5

		PosNeg_loss = 0
		IntraClass_loss = 0

		while True:
			if step % 50 == 0:
				iA,oA, iB, oB, task_id = dataloader.getBatch(5,10,update = True)
			else:
				iA,oA, iB, oB, task_id = dataloader.getBatch(5,10,update = False)
			#print(step)
			if only_interclass == False:
				if step % 2 == 0:
					#iB = iA

					ind = random.randint(0,4)

					for i in xrange(5):
						if i == ind :
							continue 
						oA[i,:,:,0] = 0

					

				ret = model.trainModel(iA,oA,iB,oB, 1.0)
				loss = ret[1]
				loss_curve = ret[2:]

			elif step % 100 == 0:
				ret = model.trainModel(iA,oA,iB,oB, 1.0)
				loss = ret[1]
				loss_curve = ret[2:]



			# IntraModel


			new_iA = np.zeros((5,image_size,image_size,4))
			new_oA = np.zeros((5,image_size,image_size,1))
			

			new_iA[:,:,:,0:3] = iA 



			if step % 50 == 0:
				rr,_,_ = model.runModel(iA,oA,iA,oA)
				new_iA[:,:,:,3] = rr[:,:,:,0]
			else:
				new_iA[:,:,:,3] = oA[:,:,:,0] # Target Output

			

			ind = random.randint(0,4)

			new_oA = oA

			for i in xrange(5):
				if i == ind :
					continue 
				new_oA[i,:,:,0] = 0

			

			ret = model.trainModelIntraClass(new_iA,new_oA,new_iA,new_oA, 1.0)

			IntraClass_loss += ret[1]


			if step % 2 == 1:
				if task_id in loss_curves:
					loss_curves[task_id][0] = [loss_curves[task_id][0][ind] + loss_curve[ind] for ind in xrange(len(loss_curve))]
					loss_curves[task_id][1] += 1
				else:
					loss_curves[task_id] = [loss_curve,1]

				if task_id in losses:
					losses[task_id][0] += loss
					losses[task_id][1] += 1

				else:
					losses[task_id] = [loss,1]
			else:
				PosNeg_loss = PosNeg_loss + loss


			step = step + 1


			if step % 10 == 0:
				print(step)
			if step % 200 == 0 or step == 1:
				train_loss = 0
				PosNeg_loss = PosNeg_loss / 100
				IntraClass_loss = IntraClass_loss / 200
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


					print(step, "total loss", s/cc, "PosNegLoss",PosNeg_loss, "IntraClass_loss",IntraClass_loss)
					train_loss = s/cc

					longterm_loss += s/cc

					if step % 1000 == 0:
						for t in xrange(5):
							task_lr[t] = 1.0

					if step % 200 == 0:
						model.saveModelAll(model_folder+"/modelAll%d"%step)


				eeid = 0

				

				# for i in xrange(5):
				# 	eeid = 0

				# 	testInputs = [testCases[0][i],testCases[1][i], testCases[2][i],testCases[3][i],i]

				# 	result, result_debug = model.runModel(testInputs[0],testInputs[1],testInputs[2])
				# 	tmp_img = np.zeros((512,512,3), dtype=np.uint8)
				# 	tmp_img2 = np.zeros((512,512), dtype=np.uint8)

				# 	for ind in xrange(len(testInputs[2])):
				# 		tmp_img = (testInputs[2][ind,:,:,:]*255).reshape(512,512,3).astype(np.uint8)
				# 		Image.fromarray(tmp_img).save(output_folder1+'img_%d_%d_sat.png' % (eeid, ind))
				# 		#if ind < 5:
				# 		#	print(np.amin(result[ind,:,:,:]*255), np.amax(result[ind,:,:,:]*255))

				# 		tmp_img2 = (result[ind,:,:,0]*255).reshape(512,512).astype(np.uint8)
				# 		Image.fromarray(tmp_img2).save(output_folder1+'img_%d_%d_task%d_output.png' % (eeid, ind, testInputs[4]))
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

				# 		Image.fromarray(tmp_img2).save(output_folder1+'img_%d_%d_task%d_output_sharp.png' % (eeid, ind, testInputs[4]))

				# 		tmp_img2 = (testInputs[3][ind,:,:,0]*255).reshape(512,512).astype(np.uint8)
				# 		Image.fromarray(tmp_img2).save(output_folder1+'img_%d_%d_task%d_target.png' % (eeid, ind, testInputs[4]))
				
				test_num = 2

				if step % 1000 == 0:
					tmpTestCase = testCase
				else:
					tmpTestCase = testCase[:test_num]

				test_loss = 0
				test_loss_first_two = 0
				if only_interclass == False:
					eeid = 0
					for testCases in tmpTestCase:
						for i in [1,3]:
							testInputs = [testCases[0][i],testCases[1][i], testCases[2][i],testCases[3][i],i]
							#print(testInputs)
							result, result_loss, result_debug = model.runModel(testInputs[0],testInputs[1],testInputs[2], testInputs[3])

							test_loss = test_loss + result_loss * dataloader.distribution[i]
							if eeid < test_num:
								test_loss_first_two = test_loss_first_two + result_loss * dataloader.distribution[i]

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

								tmp_img2 = tmp_img2.astype(np.uint8)

								Image.fromarray(tmp_img2).save(output_folder1+'img_%d_%d_task%d_output_sharp.png' % (eeid, ind, testInputs[4]))

								tmp_img2 = (testInputs[3][ind,:,:,0]*255).reshape(512,512).astype(np.uint8)
								Image.fromarray(tmp_img2).save(output_folder1+'img_%d_%d_task%d_target.png' % (eeid, ind, testInputs[4]))
								
						eeid += 1

				test_loss/=len(tmpTestCase)
				test_loss_first_two /= test_num

				print("test_loss", test_loss_first_two, test_loss)

				summary = model.addLog(test_loss_first_two, train_loss, model.meta_lr_val, PosNeg_loss,IntraClass_loss)
				writer.add_summary(summary, step)
				
				PosNeg_loss = 0
				IntraClass_loss = 0

				if only_interclass == False:
					# PosNeg Test
					eeid = 0
					for testCases in testCasePosNeg:
						for i in [1,3]:
							testInputs = [testCases[0][i],testCases[1][i], testCases[2][i],testCases[3][i],i]
							#print(testInputs)
							result, result_loss, result_debug = model.runModel(testInputs[0],testInputs[1],testInputs[2], testInputs[3])

							test_loss = test_loss + result_loss * dataloader.distribution[i]
							if eeid < test_num:
								test_loss_first_two = test_loss_first_two + result_loss * dataloader.distribution[i]

							tmp_img = np.zeros((512,512,3), dtype=np.uint8)
							tmp_img2 = np.zeros((512,512), dtype=np.uint8)

							for ind in xrange(len(testInputs[2])):
								tmp_img = (testInputs[2][ind,:,:,:]*255).reshape(512,512,3).astype(np.uint8)
								Image.fromarray(tmp_img).save(output_folder1+'img_posneg_%d_%d_sat.png' % (eeid, ind))
								#if ind < 5:
								#	print(np.amin(result[ind,:,:,:]*255), np.amax(result[ind,:,:,:]*255))

								tmp_img2 = (result[ind,:,:,0]*255).reshape(512,512).astype(np.uint8)
								Image.fromarray(tmp_img2).save(output_folder1+'img_posneg_%d_%d_task%d_output.png' % (eeid, ind, testInputs[4]))
								#if ind == 0:
								#	print(tmp_img2)

								tmp_img2 = (result[ind,:,:,0]*255).reshape(512,512)
								#tmp_img2 = (result_debug[ind,:,:,0]*255).reshape(512,512)
								tmp_img2[np.where(tmp_img2>127)] = 255
								tmp_img2[np.where(tmp_img2<=127)] = 0

								tmp_img2 = tmp_img2.astype(np.uint8)

								Image.fromarray(tmp_img2).save(output_folder1+'img_posneg_%d_%d_task%d_output_sharp.png' % (eeid, ind, testInputs[4]))

								tmp_img2 = (testInputs[3][ind,:,:,0]*255).reshape(512,512).astype(np.uint8)
								Image.fromarray(tmp_img2).save(output_folder1+'img_posneg_%d_%d_task%d_target.png' % (eeid, ind, testInputs[4]))
								
						eeid += 1

				# IntraClass Test
				eeid = 0

				new_iA = np.zeros((5,image_size,image_size,4))
				new_oA = np.zeros((5,image_size,image_size,1))
			

				for testCases in testCase:
					for i in [1,3]:
						new_iA[:,:,:,0:3] = testCases[0][i]
						new_iA[:,:,:,3] = testCases[1][i][:,:,:,0]

						ind = random.randint(0,4)
						new_oA = np.zeros((5,image_size,image_size,1))
						new_oA[ind,:,:,0] = testCases[1][i][ind,:,:,0]

						
						testInputs = [new_iA, new_oA, new_iA, new_oA,i]
						#testInputs = [testCases[0][i],testCases[1][i], testCases[2][i],testCases[3][i],i]
						#print(testInputs)

						result, result_loss = model.runModelIntraClass(testInputs[0],testInputs[1],testInputs[2], testInputs[3])

						test_loss = test_loss + result_loss * dataloader.distribution[i]
						if eeid < test_num:
							test_loss_first_two = test_loss_first_two + result_loss * dataloader.distribution[i]

						tmp_img = np.zeros((512,512,3), dtype=np.uint8)
						tmp_img2 = np.zeros((512,512), dtype=np.uint8)

						for ind in xrange(len(testInputs[2])):
							tmp_img = (testInputs[2][ind,:,:,0:3]*255).reshape(512,512,3).astype(np.uint8)
							Image.fromarray(tmp_img).save(output_folder1+'img_interclass_%d_%d_sat.png' % (eeid, ind))
							#if ind < 5:
							#	print(np.amin(result[ind,:,:,:]*255), np.amax(result[ind,:,:,:]*255))

							tmp_img2 = (result[ind,:,:,0]*255).reshape(512,512).astype(np.uint8)
							Image.fromarray(tmp_img2).save(output_folder1+'img_interclass_%d_%d_task%d_output.png' % (eeid, ind, testInputs[4]))
							#if ind == 0:
							#	print(tmp_img2)

							tmp_img2 = (result[ind,:,:,0]*255).reshape(512,512)
							#tmp_img2 = (result_debug[ind,:,:,0]*255).reshape(512,512)
							tmp_img2[np.where(tmp_img2>127)] = 255
							tmp_img2[np.where(tmp_img2<=127)] = 0

							tmp_img2 = tmp_img2.astype(np.uint8)

							Image.fromarray(tmp_img2).save(output_folder1+'img_interclass_%d_%d_task%d_output_sharp.png' % (eeid, ind, testInputs[4]))

							tmp_img2 = (testInputs[3][ind,:,:,0]*255).reshape(512,512).astype(np.uint8)
							Image.fromarray(tmp_img2).save(output_folder1+'img_interclass_%d_%d_task%d_target.png' % (eeid, ind, testInputs[4]))

							tmp_img2 = (testInputs[0][ind,:,:,3]*255).reshape(512,512).astype(np.uint8)
							Image.fromarray(tmp_img2).save(output_folder1+'img_interclass_%d_%d_task%d_input.png' % (eeid, ind, testInputs[4]))
							
					eeid += 1









				if step % 5000 == 0:

					print("longterm_loss",longterm_loss/5," lr is", model.meta_lr_val)

					longterm_loss = test_loss

					print(longterm_loss, last_longterm_loss)

					if longterm_loss > last_longterm_loss*1.0 and step > cool_down:
						#reduce learning rate
						#if model.meta_lr_val > 0.0001:
						if model.meta_lr_val > 0.000005:
							model.meta_lr_val = model.meta_lr_val / 2
							longterm_loss = 10000 # don't reduce the learning rate two times

							cool_down = step + 40000
						# new_lr = sess.run([model.change_lr_op], feed_dict = {model.meta_lr_val:model.meta_lr_val})

						print("change lr to", model.meta_lr_val)

					last_longterm_loss = longterm_loss
					longterm_loss = 0




				


				t_int = time() - ts 
				ts = time()

				print("iterations per hour", 3600/t_int * 200, t_int, dataloader.total_time, dataloader.total_time/t_int)
				dataloader.total_time = 0

	




		writer.close()





