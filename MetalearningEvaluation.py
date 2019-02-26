import argparse

from MetalearningModels import MAMLFirstOrder
import MetalearningLoader 
from time import time,sleep
from subprocess import Popen
import random
import scipy
import scipy.ndimage
from PIL import Image
import os, datetime
import numpy as np
import tensorflow as tf

import json 


parser = argparse.ArgumentParser()


parser.add_argument('-model', action='store', dest='model_type', type=str,
                    help='model type')

parser.add_argument('-r', action='store', dest='model_recover', type=str,
                    help='path to model for recovering',required=True)

parser.add_argument('-d', action='store', dest='data_folder', type=str,
                    help='dataset folder', default='/data/songtao/')



args = parser.parse_args()

print(args)

class BenchmarkTestCase():
	def __init__(self, model, model_reset_func, folder_name, number_of_image = 2, train_iteration=20, train_lr = 0.001, name = "default"):
		self.sat_imgs = []
		self.target_imgs = []
		self.outputs = []
		self.model = model 
		self.model_reset_func = model_reset_func
		self.train_iteration = train_iteration
		self.train_lr = train_lr
		self.num_of_images = number_of_image 
		self.test_name = name 

		for i in xrange(number_of_image):
			try:
				sat_img = scipy.ndimage.imread(folder_name+"/sat%d.png"%i).astype(np.float)/256.0 # 256.0  be consistent with dataloader20180429
			except:
				try:
					sat_img = scipy.ndimage.imread(folder_name+"/sat%d.jpg"%i).astype(np.float)/256.0 # 256.0  be consistent with dataloader20180429
				except:
					print("can't find ", folder_name+"/sat%d.jpg/png"%i)
					exit() 

			try:
				target_img = scipy.ndimage.imread(folder_name+"/target%d.png"%i).astype(np.float)/256.0 # 256.0  be consistent with dataloader20180429
			except:
				try:
					target_img = scipy.ndimage.imread(folder_name+"/target%d.jpg"%i).astype(np.float)/256.0 # 256.0  be consistent with dataloader20180429
				except:
					print("can't find ", folder_name+"/target%d.jpg/png"%i)
					exit() 

			self.sat_imgs.append(sat_img[:,:,0:3])
			self.target_imgs.append(target_img[:,:,0])


	def _reset(self):
		self.model_reset_func(0) 



	def _run_on_image(self, img):

		dimx = np.shape(img)[0]
		dimy = np.shape(img)[1]

		output_dim = ((dimx/256+1)*256, (dimy/256+1)*256)
		batch_size = (dimx/256+1) * (dimy/256+1)
		input_frame_pad = np.zeros(((dimx/256+1)*256, (dimy/256+1)*256, 3))
		result_pad = np.zeros(((dimx/256+1)*256, (dimy/256+1)*256))

		test_inputB = np.zeros((batch_size,512,512,3))
		test_outputB= np.zeros((batch_size,512,512,1))
		result= np.zeros((batch_size,512,512,2))


		input_frame_pad[0:dimx,0:dimy,:] = img 

		ind = 0
		for x in xrange(output_dim[0]/256-1):
			for y in xrange(output_dim[1]/256-1):
				
				test_inputB[ind,:,:,:] = input_frame_pad[x*256:x*256+512, y*256:y*256+512,:]

				ind += 1


		for i in xrange(0, ind/32+1):
			#print(i)
			b = min(32,ind - i*32)
			if b == 0:
				continue
			result_, _= self.model.runBaselineModel(test_inputB[i*32:i*32+b,:,:,:],test_outputB[i*32:i*32+b,:,:,:])
			result[i*32:i*32+b,:,:,:] = result_


		ind = 0
		for x in xrange(output_dim[0]/256-1):
			for y in xrange(output_dim[1]/256-1):
				
				xl = 128
				xr = 384
				yl = 128
				yr = 384

				if x == 0:
					xl = 0
				if x == output_dim[0]/256-2:
					xr = 512

				if y == 0:
					yl = 0

				if y == output_dim[1]/256-2:
					yr = 512

				result_pad[x*256+xl:x*256+xr, y*256+yl:y*256+yr] =  result[ind,xl:xr,yl:yr,0]				
				ind += 1

		return result_pad[0:dimx,0:dimy] 



	def _default_sampler(self, img_ids = [0], example_sample = 20):
		
		def sampler():
			example_inputA = np.zeros((example_sample, 512,512,3))
			example_targetA = np.zeros((example_sample, 512, 512, 1))
			for i in xrange(example_sample):
				while True:
					img_id = np.random.choice(img_ids)
					example_sat = self.sat_imgs[img_id]
					example_target = self.target_imgs[img_id]

					x = random.randint(0, np.shape(example_sat)[0] - 512)
					y = random.randint(0, np.shape(example_sat)[1] - 512)

					example_inputA_crop = np.copy(example_sat[x:x+512,y:y+512,0:3])
					example_targetA_crop = np.copy(example_target[x:x+512,y:y+512])

					t = random.randint(0,4)
					
					if t == 1:
						example_inputA_crop = np.flipud(example_inputA_crop)
						example_targetA_crop = np.flipud(example_targetA_crop)

					if t == 2:
						example_inputA_crop  = np.fliplr(example_inputA_crop)
						example_targetA_crop = np.fliplr(example_targetA_crop)

					if t == 3:
						example_inputA_crop  = np.rot90(example_inputA_crop,k=1,axes=[0,1])
						example_targetA_crop = np.rot90(example_targetA_crop,k=1,axes=[0,1])

					if t == 4:
						example_inputA_crop  = np.rot90(example_inputA_crop ,k=3,axes=[0,1])
						example_targetA_crop = np.rot90(example_targetA_crop,k=3,axes=[0,1])

					example_inputA[i, :,:,:] = example_inputA_crop
					example_targetA[i,:,:,0] = example_targetA_crop

					if np.sum(example_targetA_crop)>5.0 or i>example_sample*0.6 :
						break

					cc = cc + 1
					if cc > 10:
						break

			return example_inputA, example_targetA

		return sampler 


	def _train(self, sampler):

		lr_scale = 1.0 
		for i in xrange(self.train_iteration+1):
			example_inputA,example_targetA = sampler()
			_, loss = self.model.trainBaselineModel(example_inputA,example_targetA, scale = lr_scale)
			lr_scale *= 0.9
			if i % 10 == 0:
				print(i,loss)


	def _metric(self, output, target, threshold = 0.5):

		I = len(np.where((output>=threshold) & (target>=threshold))[0])
		U = len(np.where((output>=threshold) | (target>=threshold))[0])
		G = len(np.where((target>=threshold))[0])
		O = len(np.where((output>=threshold))[0])

		# print("IOU:", float(I)/float(U))
		# print("Precision:", float(I)/float(O))
		# print("Recall:", float(I)/float(G))


		return float(I)/float(U+1), float(I)/float(O+1), float(I)/float(G+1)


	def EvaluateOneShot(self): #use one example
		self._reset() 
		self._train(self._default_sampler()) 

		result = [0] * 3 

		for i in xrange(self.num_of_images):
			output = self._run_on_image(self.sat_imgs[i])

			iou, precision, recall = self._metric(output, self.target_imgs[i])

			result[0] += iou 
			result[1] += precision 
			result[2] += recall 

		result = [x / self.num_of_images for x in result]


		print("[%s]\t IOU %.3f \t Precision %.3f \t Recall %.3f" % (self.test_name, result[0], result[1], result[2]))

		return result




class Benchmark():
	def __init__(self, model, model_reset_func, config = None, train_iteration=20, train_lr = 0.001):
		if config is None:
			config = {}
			config['folder'] = "benchmark/"
			config['test_cases'] = []

			config['test_cases'].append(["building_flood",2])

		self.config = config 
		self.model = model 
		self.model_reset_func = model_reset_func	
		self.train_iteration = train_iteration 
		self.train_lr = train_lr 		

	def Evaluate(self):
		result = [0]*3
		for test_case in self.config['test_cases']:
			test_case_object = BenchmarkTestCase(self.model, self.model_reset_func, self.config['folder']+test_case[0], number_of_image=test_case[1], train_iteration=self.train_iteration, train_lr = self.train_lr, name = test_case[0])

			iou, precision, recall = test_case_object.EvaluateOneShot()

			result[0] += iou 
			result[1] += precision 
			result[2] += recall 

		result = [x / len(self.config['test_cases']) for x in result]

		print("[Average]\t IOU %.3f \t Precision %.3f \t Recall %.3f" % (result[0], result[1], result[2]))


if __name__ == "__main__":

	with tf.Session() as sess:
		model = MAMLFirstOrder(sess)
		model.restoreModel(args.model_recover)

		benchmark = Benchmark(model, lambda _: model.restoreModel(args.model_recover), train_iteration=40)  
		benchmark.Evaluate()




