import os, datetime
import numpy as np
from PIL import Image
import random
import scipy
from time import time
import sys
from subprocess import Popen
import scipy.misc
import scipy.ndimage as nd 



# def FastSample(mask_img, crop_size, num = 5, force_positive=False):
# 	#print(num)
# 	img_sx = np.shape(mask_img)[0]
# 	img_sy = np.shape(mask_img)[1]

# 	output_list = []

# 	if force_positive == False:
# 		for i in xrange(num):
# 			rx = random.randint(0, img_sx - crop_size-1)
# 			ry = random.randint(0, img_sy - crop_size-1)
# 			output_list.append((rx,ry))
# 	else:
# 		scale = 32

# 		new_mask = scipy.misc.imresize(mask_img, (img_sx/scale,img_sy/scale))
# 		#print(np.shape(new_mask))
# 		sum_mask = nd.convolve(new_mask.astype(float),np.ones((6,6)),mode='constant',cval=0.0)

# 		if np.sum(sum_mask) < 1.0:
# 			return [] 

# 		for i in xrange(num):
# 			cc = 0
# 			while True:
# 				rx = random.randint(0, img_sx - crop_size-1)
# 				ry = random.randint(0, img_sy - crop_size-1)

# 				if sum_mask[(rx+crop_size/2)/scale, (ry+crop_size/2)/scale] > 0:
# 					output_list.append((rx,ry))
# 					break
# 				else:
# 					cc = cc + 1
# 					if cc > 30: # only try 30 times
# 						output_list.append((rx,ry))
# 						break

# 	return output_list

def FastSample(mask_img, crop_size, num = 5, force_positive=False):
	#print(num)
	img_sx = np.shape(mask_img)[0]
	img_sy = np.shape(mask_img)[1]

	output_list = []

	if force_positive == False:
		for i in xrange(num):
			rx = random.randint(0, img_sx - crop_size-1)
			ry = random.randint(0, img_sy - crop_size-1)
			output_list.append((rx,ry))
	else:
		scale = 32

		new_mask = scipy.misc.imresize(mask_img, (img_sx/scale,img_sy/scale))
		#print(np.shape(new_mask))
		sum_mask = nd.convolve(new_mask.astype(float),np.ones((6,6)),mode='constant',cval=0.0)

		if np.sum(sum_mask) < 1.0:
			return [] 

		for i in xrange(num):
			cc = 0
			while True:
				rx = random.randint(0, img_sx - crop_size-1)
				ry = random.randint(0, img_sy - crop_size-1)

				if sum_mask[(rx+crop_size/2)/scale, (ry+crop_size/2)/scale] > 0:
					output_list.append((rx,ry))
					break
				else:
					cc = cc + 1
					if cc > 30: # only try 30 times
						output_list.append((rx,ry))
						break

	return output_list


#class Loader(object):
# Implement data loader for different tasks 
class LoaderOSM(object):
	def __init__ (self, folders, region_n = 36, subtask=0):
		# 2 is road
		# 4 is building 
		self.taskid = subtask
		self.folders = folders
		self.region_n = region_n

	def load(self, n, imagesize, p_ratio = 0.8, sample_per_region=10):

		batch_n = n/sample_per_region
		output = []

		for i in xrange(batch_n):
			#load mask 
			while True:
				rid = random.randint(0,self.region_n-1)
				did = random.randint(0,3)
				folder = np.random.choice(self.folders)

				try:
					mask = scipy.ndimage.imread(folder+"/region%d_%d_t%d.png"%(rid,did,self.taskid))
				except:
					#print("Invalid Region",rid,did,folder,self.taskid)
					continue 

				coord_list = []
				l = FastSample(mask, imagesize, int(sample_per_region*p_ratio),True)

				if len(l)==0:
					#print("Invalid Region (0)",rid,did,folder,self.taskid)
					continue 

				coord_list = coord_list + l 
				coord_list = coord_list + FastSample(mask, imagesize, sample_per_region-int(sample_per_region*(p_ratio)),False)

				sat = scipy.ndimage.imread(folder+"/region%d_%d_sat.png"%(rid,did))

				#print(coord_list)
				for j in xrange(sample_per_region):
					sat_img = np.zeros((imagesize,imagesize, 3),dtype=float)
					mask_img = np.zeros((imagesize,imagesize, 1),dtype=float)

					ix = coord_list[j][0]
					iy = coord_list[j][1]

					sat_img = sat[ix:ix+imagesize, iy:iy+imagesize,:]
					mask_img[:,:,0] = mask[ix:ix+imagesize, iy:iy+imagesize]

					sat_img = sat_img/256.0
					mask_img = mask_img/256.0

					output.append((sat_img,mask_img))
				break
		return output

class LoaderKaggleDSTL(object):
	def __init__ (self, folder, subtask=0): 
		self.folder = folder
		self.taskid = subtask
		filelist = os.listdir(self.folder)

		self.filelist = []

		for f in filelist:
			if f.endswith("sat.png"):
				items = f.split('_')
				self.filelist.append(items[0]+"_"+items[1]+"_"+items[2])

	def load(self, n, imagesize, p_ratio = 0.8, sample_per_region=5):

		batch_n = n/sample_per_region
		output = []

		for i in xrange(batch_n):
			#load mask 
			while True:
				f = np.random.choice(self.filelist)

				try:
					mask = scipy.ndimage.imread(self.folder+"/%s_task%d.png"%(f,self.taskid))[:,:]
				except:
					#print("Invalid Region",self.folder+"/%s_task%d.png"%(f,self.taskid))
					continue 

				coord_list = []
				l = FastSample(mask, imagesize, int(sample_per_region*p_ratio),True)

				if len(l)==0:
					#print("Invalid Region (0)",self.folder+"/%s_task%d.png"%(f,self.taskid))
					continue 

				coord_list = coord_list + l 
				coord_list = coord_list + FastSample(mask, imagesize, sample_per_region-int(sample_per_region*(p_ratio)),False)

				sat = scipy.ndimage.imread(self.folder+"/%s_sat.png"%f)

				for j in xrange(sample_per_region):
					sat_img = np.zeros((imagesize,imagesize, 3),dtype=float)
					mask_img = np.zeros((imagesize,imagesize, 1),dtype=float)

					ix = coord_list[j][0]
					iy = coord_list[j][1]

					sat_img = sat[ix:ix+imagesize, iy:iy+imagesize,:]
					mask_img[:,:,0] = mask[ix:ix+imagesize, iy:iy+imagesize]

					sat_img = sat_img/256.0
					mask_img = mask_img/256.0

					output.append((sat_img,mask_img))
				break
		return output	

class LoaderCVPRContestLandCover(object):
	def __init__ (self, folder, subtask=0):
		self.taskid = subtask
		self.folder = folder

		filelist = os.listdir(self.folder)

		self.filelist = []

		for f in filelist:
			if f.endswith("sat.jpg"):
				self.filelist.append(f.split('_')[0])

	def load(self, n, imagesize, p_ratio = 0.5, sample_per_region=10):

		batch_n = n/sample_per_region
		output = []

		for i in xrange(batch_n):
			#load mask 
			while True:
				f = np.random.choice(self.filelist)

				try:
					mask = scipy.ndimage.imread(self.folder+"/%s_mask_%d.png"%(f,self.taskid))
				except:
					#print("Invalid Region",self.folder+"/%s_mask_%d.png"%(f,self.taskid))
					continue 

				
				l = FastSample(mask, imagesize, int(sample_per_region*p_ratio),True)

				if len(l)==0:
					#print("Invalid Region (0)",self.folder+"/%s_mask_%d.png"%(f,self.taskid))
					continue 
				coord_list = []
				coord_list = coord_list + l 
				#coord_list = coord_list + FastSample(mask, imagesize, sample_per_region-int(sample_per_region*(p_ratio)),False)

				sat = scipy.ndimage.imread(self.folder+"/%s_sat.jpg"%f)

				for j in xrange(int(sample_per_region*p_ratio)):
					sat_img = np.zeros((imagesize,imagesize, 3),dtype=float)
					mask_img = np.zeros((imagesize,imagesize, 1),dtype=float)

					ix = coord_list[j][0]
					iy = coord_list[j][1]

					sat_img = sat[ix:ix+imagesize, iy:iy+imagesize,:]
					mask_img[:,:,0] = mask[ix:ix+imagesize, iy:iy+imagesize]

					sat_img = sat_img/256.0
					mask_img = mask_img/256.0

					output.append((sat_img,mask_img))
				break

			while True:
				f = np.random.choice(self.filelist)

				try:
					mask = scipy.ndimage.imread(self.folder+"/%s_mask_%d.png"%(f,self.taskid))
					mask = 255-mask
				except:
					#print("Invalid Region",self.folder+"/%s_mask_%d.png"%(f,self.taskid))
					continue 

				
				l = FastSample(mask, imagesize, sample_per_region-int(sample_per_region*p_ratio),True)

				if len(l)==0:
					#print("Invalid Region (0)",self.folder+"/%s_mask_%d.png"%(f,self.taskid))
					continue 
				coord_list = []
				coord_list = coord_list + l 
				#coord_list = coord_list + FastSample(mask, imagesize, sample_per_region-int(sample_per_region*(p_ratio)),False)

				sat = scipy.ndimage.imread(self.folder+"/%s_sat.jpg"%f)
				mask = 255-mask


				for j in xrange(sample_per_region-int(sample_per_region*p_ratio)):
					sat_img = np.zeros((imagesize,imagesize, 3),dtype=float)
					mask_img = np.zeros((imagesize,imagesize, 1),dtype=float)

					ix = coord_list[j][0]
					iy = coord_list[j][1]

					sat_img = sat[ix:ix+imagesize, iy:iy+imagesize,:]
					mask_img[:,:,0] = mask[ix:ix+imagesize, iy:iy+imagesize]

					sat_img = sat_img/256.0
					mask_img = mask_img/256.0

					output.append((sat_img,mask_img))
				break



		return output

class LoaderCVPRContestRoadDetection(object):
	def __init__ (self, folder):
		self.folder = folder

		filelist = os.listdir(self.folder)

		self.filelist = []

		for f in filelist:
			if f.endswith("sat.jpg"):
				self.filelist.append(f.split('_')[0])

	def load(self, n, imagesize, p_ratio = 0.8, sample_per_region=5):

		batch_n = n/sample_per_region
		output = []

		for i in xrange(batch_n):
			#load mask 
			while True:
				f = np.random.choice(self.filelist)

				try:
					mask = scipy.ndimage.imread(self.folder+"/%s_mask.png"%f)[:,:,0]
				except:
					#print("Invalid Region",self.folder+"/%s_mask.png"%f)
					continue 

				coord_list = []
				l = FastSample(mask, imagesize, int(sample_per_region*p_ratio),True)

				if len(l)==0:
					#print("Invalid Region (0)",self.folder+"/%s_mask.png"%f)
					continue 

				coord_list = coord_list + l 
				coord_list = coord_list + FastSample(mask, imagesize, sample_per_region-int(sample_per_region*(p_ratio)),False)

				sat = scipy.ndimage.imread(self.folder+"/%s_sat.jpg"%f)

				for j in xrange(sample_per_region):
					sat_img = np.zeros((imagesize,imagesize, 3),dtype=float)
					mask_img = np.zeros((imagesize,imagesize, 1),dtype=float)

					ix = coord_list[j][0]
					iy = coord_list[j][1]

					sat_img = sat[ix:ix+imagesize, iy:iy+imagesize,:]
					mask_img[:,:,0] = mask[ix:ix+imagesize, iy:iy+imagesize]

					sat_img = sat_img/256.0
					mask_img = mask_img/256.0

					output.append((sat_img,mask_img))
				break
		return output




class DataLoaderMultiplyTask(object):
	def __init__ (self, loaders, imagesize=512):
		self.loaders = loaders # a list of loading functions
		self.imagesize = imagesize
		self.tid = 0 
		self.cc = 0

	def preload(self, num_per_task=100):
		self.preloadData = []

		for i in xrange(len(self.loaders)):
			self.preloadData.append(self.loaders[i].load(num_per_task,self.imagesize))

	def preload_detail(self, num_per_task):
		self.preloadData = []

		for i in xrange(len(self.loaders)):
			self.preloadData.append(self.loaders[i].load(num_per_task[i],self.imagesize))


	def loadBatchFromTask(self, taskid, sizeA=5, sizeB=10):
		inputA = np.zeros((sizeA,self.imagesize, self.imagesize, 3))
		outputA = np.zeros((sizeA,self.imagesize, self.imagesize, 1))
		inputB = np.zeros((sizeB,self.imagesize, self.imagesize, 3))
		outputB = np.zeros((sizeB,self.imagesize, self.imagesize, 1))

		n = len(self.preloadData[taskid])

		for i in xrange(sizeA):
			ind = random.randint(0,n-1)

			inputA[i,:,:,:] = self.preloadData[taskid][ind][0]
			outputA[i,:,:,:] = self.preloadData[taskid][ind][1]

		for i in xrange(sizeB):
			ind = random.randint(0,n-1)

			inputB[i,:,:,:] = self.preloadData[taskid][ind][0]
			outputB[i,:,:,:] = self.preloadData[taskid][ind][1]

		return inputA, outputA, inputB, outputB

	def loadBatch(self, sizeA=5, sizeB=10, p=None):
		if self.cc % 2 == 0:
			if p is None:
				tid = random.randint(0,len(self.loaders)-1)
			else:
				tid = np.random.choice(range(len(self.loaders)), p=p)
			self.tid = tid 
		else:
			tid = self.tid 

		self.cc += 1


		ret = self.loadBatchFromTask(tid,sizeA=sizeA, sizeB=sizeB)

		return ret[0], ret[1], ret[2], ret[3], tid 




