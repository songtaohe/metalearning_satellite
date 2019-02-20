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
from subprocess import Popen




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

		if preload == False:
			self.distribution = dist
			return 



		input_imgs = []
		target_imgs = []

		for i in xrange(region_num):
			target_imgs.append([])
			input_imgs.append(scipy.ndimage.imread(foldername + "/"+"region%d_sat.png"%i).astype(np.float)/255.0)
			for j in xrange(task_num):
				if j == baseline_task_id:
					target_imgs[-1].append(scipy.ndimage.imread(foldername + "/"+"region%d_t%d.png"%(i,j+1)).astype(np.float)/255.0)
					self.distribution[j] += np.sum(target_imgs[-1][-1], axis =(0,1)) / (4096*4096)

				else:
					target_imgs[-1].append([0])
					self.distribution[j] += 1.0

				


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

		self.task_id = baseline_task_id 
		task_id = baseline_task_id 

		#task_id = random.randint(0, self.task_num-1)

		# task_id = random.randint(0, 1)

		# if task_id == 0:
		# 	task_id = 1
		# else:
		# 	task_id = 3

		if self.preload == True:
			

			retry_c = 0

			# Sample A
			for i in xrange(sizeA):
				while True:
					region_id = random.randint(0, self.region_num - 1)
					if region_id in self.bad_regions:
						continue 


					x = random.randint(0, self.region_size-image_size-1)
					y = random.randint(0, self.region_size-image_size-1)

					v_sum = np.sum(self.target_imgs[region_id][task_id][x:x+512,y:y+512])
					if v_sum > 512*512*0.05*0.05:
						break

					retry_c += 1
					if retry_c > 100:
						self.bad_regions.append(region_id)
						print("bad_regions",self.bad_regions)
						break


				inputA[i, :,:,:] = self.input_imgs[region_id][x:x+512,y:y+512,:]
				outputA[i,:,:,0] = self.target_imgs[region_id][task_id][x:x+512,y:y+512]

				#	break
			retry_c = 0
			# Sample B
			for i in xrange(sizeB):
				while True:
					region_id = random.randint(0, self.region_num - 1)
					if region_id in self.bad_regions:
						continue 
					x = random.randint(0, self.region_size-image_size-1)
					y = random.randint(0, self.region_size-image_size-1)

					v_sum = np.sum(self.target_imgs[region_id][task_id][x:x+512,y:y+512])
					if v_sum > 512*512*0.05*0.05:
						break

					retry_c += 1
					if retry_c > 100:
						self.bad_regions.append(region_id)
						print("bad_regions",self.bad_regions)
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
				for i in xrange(sizeB):
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
		
