

import sys
import tensorflow as tf
from PIL import Image
import scipy
import random
import numpy as np 
from time import time 

#isBaseLine = True
isBaseLine = False

def Test(example, test_image,target_image, model,  example_sample = 20, color_channel = 1, color_intensity = 0.5,  output_folder = "", threshold = 0.5):
	print(isBaseLine)
	example_sat = scipy.ndimage.imread(example['sat']).astype(np.float)/255.0  
	example_sat = example_sat[:,:,0:3]
	example_target = scipy.ndimage.imread(example['target']).astype(np.float)/255.0

	if len(np.shape(example_target)) == 3:
		example_target = example_target[:,:,0].reshape((4096,4096))





	example_region = example['region']

	test_img = scipy.ndimage.imread(test_image).astype(np.float)/255.0
	test_img = test_img[:,:,0:3]
	target_img = scipy.ndimage.imread(target_image).astype(np.float)/255.0

	example_sat_sample = np.copy(example_sat)

	example_inputA = np.zeros((example_sample, 512,512,3))
	example_targetA = np.zeros((example_sample, 512, 512, 1))

	for i in xrange(example_sample):
		while True:
			x = random.randint(example_region[0],example_region[2]-512)
			y = random.randint(example_region[1],example_region[3]-512)

			example_inputA[i, :,:,:] = example_sat[x:x+512,y:y+512,:]
			example_targetA[i,:,:,0] = example_target[x:x+512,y:y+512]

			v_sum = np.sum(example_targetA[i,:,:,0])
			if v_sum > 512*512*0.10*0.10:
				break


		Image.fromarray((example_inputA[i, :,:,:]*255).astype(np.uint8)).save(output_folder+"example_%d_sat.png"%i)
		Image.fromarray((example_targetA[i,:,:,0]*255).astype(np.uint8)).save(output_folder+"example_%d_target.png"%i)

		example_sat_sample[x:x+512,y:y+512,:] = example_sat_sample[x:x+512,y:y+512,:] + 0.1 


	# Train the model
	model.meta_lr_val = 0.0001

	it = 101
	if isBaseLine == False:
		it = 61
		model.meta_lr_val = 0.0001


	for i in xrange(it):
		_, loss = model.trainBaselineModel(example_inputA,example_targetA)
		if i % 20 == 0:
			print(i,loss)
	# model.meta_lr_val = 0.00001
	# for i in xrange(301,601):
	# 	_, loss = model.trainBaselineModel(example_inputA,example_targetA)
	# 	if i % 20 == 0:
	# 		print(i,loss)


	test_result = np.zeros((4096, 4096))
	test_inputB = np.zeros((16,512,512,3))
	test_outputB= np.zeros((16,512,512,1))

	# for x in xrange(0,8,2):
	# 	print(x)
	# 	for y in xrange(8):
	# 		test_inputB[y,:,:,:] = test_img[x*512:x*512+512, y*512:y*512+512,:]

	# 	for y in xrange(8):
	# 		test_inputB[y+8,:,:,:] = test_img[(x+1)*512:(x+1)*512+512, y*512:y*512+512,:]
	# 	#if isBaseLine == True:
	# 	result, _= model.runBaselineModel(test_inputB,test_outputB)
	# 	#else:
	# 	#	result, _, _ = model.runModel(example_inputA,example_targetA,test_inputB,test_outputB)
		
	# 	for y in xrange(8):
	# 		test_result[x*512:x*512+512, y*512:y*512+512] = result[y,:,:,0]
	# 	for y in xrange(8):
	# 		test_result[(x+1)*512:(x+1)*512+512, y*512:y*512+512] = result[y+8,:,:,0]
	for x in xrange(0,15):
		print(x)
		for y in xrange(15):
			test_inputB[y,:,:,:] = test_img[x*256:x*256+512, y*256:y*256+512,:]
		
		#if isBaseLine == True:
		result, _= model.runBaselineModel(test_inputB,test_outputB)
		#else:
		#	result, _, _ = model.runModel(example_inputA,example_targetA,test_inputB,test_outputB)
		for y in xrange(15):
			test_result[x*256+128:x*256+384, y*256+128:y*256+384] = result[y,128:384,128:384,0]



	test_result2 = np.copy(test_result)
	test_result3 = np.copy(test_result)

	test_img[:,:,color_channel] = np.clip(test_img[:,:,color_channel] + test_result[:,:]*color_intensity,0.0,1.0)
	Image.fromarray((test_img*255).astype(np.uint8)).save(output_folder+"test_output_soft.png")

	test_img = scipy.ndimage.imread(test_image).astype(np.float)/255.0


	print(np.amax(test_result2), np.amin(test_result2))
	threshold = (np.amax(test_result2) + np.amin(test_result2)) / 2
	print("Threshold", threshold)
	#test_result2[np.where(test_result2>threshold)] = 1.0
	test_result2[np.where(test_result2<=threshold)] = 0.0


	I = len(np.where((test_result2>=threshold) & (target_img>=threshold))[0])
	U = len(np.where((test_result2>=threshold) | (target_img>=threshold))[0])
	G = len(np.where((target_img>=threshold))[0])
	O = len(np.where((test_result2>=threshold))[0])

	print("IOU:", float(I)/float(U))
	print("Precision:", float(I)/float(O))
	print("Recall:", float(I)/float(G))





	test_img[:,:,color_channel] = np.clip(test_img[:,:,color_channel] + test_result2[:,:]*color_intensity,0.0,1.0)
	Image.fromarray((test_img*255).astype(np.uint8)).save(output_folder+"test_output_sharp.png")

	example_sat2 = np.copy(example_sat)
	example_sat2[example_region[0]:example_region[2],example_region[1]:example_region[3]] += 0.1
	example_sat2 = np.clip(example_sat2, 0, 1)
	Image.fromarray((example_sat2*255).astype(np.uint8)).save(output_folder+"example_region_mask.png")
	
	example_sat_sample = np.clip(example_sat_sample, 0, 1)
	Image.fromarray((example_sat_sample*255).astype(np.uint8)).save(output_folder+"example_region_sample.png")


	#example_sat[:,:,color_channel] = np.clip(example_sat[:,:,color_channel] + example_target[:,:]*0.5,0.0,1.0)
	example_sat[example_region[0]:example_region[2],example_region[1]:example_region[3],color_channel] = np.clip(example_sat[example_region[0]:example_region[2],example_region[1]:example_region[3],color_channel] + example_target[example_region[0]:example_region[2],example_region[1]:example_region[3]]*0.5,0.0,1.0)



	
	Image.fromarray((example_sat*255).astype(np.uint8)).save(output_folder+"example_region.png")
	Image.fromarray((test_result*255).astype(np.uint8)).save(output_folder+"test_output_bw.png")


	return test_result 

#dataset/boston_task5_small/region12_sat.png

def TestIOU(example, test_image,target_image, model,  example_sample = 20, color_channel = 1, color_intensity = 0.5,  output_folder = "", threshold = 0.5, image_size = 4096):
	print(isBaseLine)
	example_sat = scipy.ndimage.imread(example['sat']).astype(np.float)/255.0  
	example_sat = example_sat[:,:,0:3]
	example_target = scipy.ndimage.imread(example['target']).astype(np.float)/255.0

	if len(np.shape(example_target)) == 3:
		example_target = example_target[:,:,0].reshape((4096,4096))





	example_region = example['region']

	test_img = scipy.ndimage.imread(test_image).astype(np.float)/255.0
	test_img = test_img[:,:,0:3]
	target_img = scipy.ndimage.imread(target_image).astype(np.float)/255.0

	example_sat_sample = np.copy(example_sat)

	example_inputA = np.zeros((example_sample, 512,512,3))
	example_targetA = np.zeros((example_sample, 512, 512, 1))

	

	# Train the model
	model.meta_lr_val = 0.0001

	# it = 101
	# if isBaseLine == False:
	# 	it = 61
	# 	model.meta_lr_val = 0.0001

	it = 101

	IOUs = []
	ts = time()
	for i in xrange(it):
		loss = 0.0
		if i > 0:
			for ii in xrange(example_sample):
				while True:
					x = random.randint(example_region[0],example_region[2]-512)
					y = random.randint(example_region[1],example_region[3]-512)

					example_inputA[ii, :,:,:] = example_sat[x:x+512,y:y+512,:]
					example_targetA[ii,:,:,0] = example_target[x:x+512,y:y+512]

					v_sum = np.sum(example_targetA[ii,:,:,0])
					if v_sum > 512*512*0.10*0.10:
						break

			_, loss = model.trainBaselineModel(example_inputA,example_targetA)


		if i % 10 == 0:
			print(i,loss)
	
			test_img = scipy.ndimage.imread(test_image).astype(np.float)/255.0

			test_result = np.zeros((image_size, image_size))
			test_inputB = np.zeros((image_size/256,512,512,3))
			test_outputB= np.zeros((image_size/256,512,512,1))

			for x in xrange(0,image_size/256-1):
				if x % 8 == 0:
					print(x)
				for y in xrange(image_size/256-1):
					#print(y*256, y*256+512, np.shape(test_img))
					test_inputB[y,:,:,:] = test_img[x*256:x*256+512, y*256:y*256+512,0:3]
				
				#if isBaseLine == True:
				result, _= model.runBaselineModel(test_inputB,test_outputB)
				#else:
				#	result, _, _ = model.runModel(example_inputA,example_targetA,test_inputB,test_outputB)
				for y in xrange(image_size/256-1):
					test_result[x*256+128:x*256+384, y*256+128:y*256+384] = result[y,128:384,128:384,0]


			test_result2 = test_result
			

			print(np.amax(test_result2), np.amin(test_result2))
			threshold = (np.amax(test_result2) + np.amin(test_result2)) / 2
			print("Threshold", threshold)
			#test_result2[np.where(test_result2>threshold)] = 1.0
			test_result2[np.where(test_result2<=threshold)] = 0.0

			test_img[:,:,color_channel] = np.clip(test_img[:,:,color_channel] + test_result2[:,:]*color_intensity,0.0,1.0)
			
			Image.fromarray((test_img*255).astype(np.uint8)).save(output_folder+"test_output_sharp_%d.png"%i)



			I = len(np.where((test_result2>=threshold) & (target_img>=threshold))[0])
			U = len(np.where((test_result2>=threshold) | (target_img>=threshold))[0])
			G = len(np.where((target_img>=threshold))[0])
			O = len(np.where((test_result2>=threshold))[0])

			print("IOU:", float(I)/float(U))
			print("Precision:", float(I)/float(O))
			print("Recall:", float(I)/float(G))

			IOUs.append(float(I)/float(U))


			print("Time left",(time()-ts)*(it-i)/10)
			ts = time()

	print(IOUs)


	return test_result 

if __name__ == "__main__":
	if sys.argv[7] == "M":
		from MAMLV1_20180406 import *
		isBaseLine = False
	else:
		from MAMLV1_20180406_baseline_road_inputFilter import *
		isBaseLine = True

# load model
# test on examples 
	#random.seed(int(time()))
	random.seed(123)

	# example = {"sat":"/data/songtao/metalearning/dataset/boston_task5_small/region12_sat.png", 
	# 		"target":"/data/songtao/metalearning/dataset/boston_task5_small/region12_t2.png",
	# 		"region":[200,200,1600,1600]}

	# example = {"sat":"/data/songtao/metalearning/dataset/boston_task5_small/region12_sat.png", 
	# 		"target":"/data/songtao/metalearning/dataset/boston_task5_small/region12_t4.png",
	# 		"region":[200,200,1600,1600]}

	# example = {"sat":"/data/songtao/metalearning/dataset/boston_task5_small/region12_sat.png", 
	# 		"target":"/data/songtao/metalearning/example/example_region_mask_car.png",
	# 		"region":[200,200,1600,1600]}

	# example = {"sat":"/data/songtao/metalearning/dataset/boston_task5_small/region12_sat.png", 
	# 		"target":"/data/songtao/metalearning/example/example_region_mask_tree.png",
	# 		"region":[200,200,1600,1600]}

	# example = {"sat":"/data/songtao/metalearning/dataset/boston_task5_small/region12_sat.png", 
	# 		"target":"/data/songtao/metalearning/example/example_region_mask_grass.png",
	# 		"region":[200,200,1600,1600]}

	example = {"sat":sys.argv[2], 
			"target":sys.argv[3],
			"region":[2550,450,2550+550,450+550]}
			#"region":[450,2550,450+550,2550+550]}
			#"region":[2048,2048,4095-1024,4095-1024]}
			#"region":[0,3072,1024,4095]} # top-right corner
			#"region":[2048,0,4095,2047]} # bottom-left 1km by 1km region
			#"region":[3072,0,4095,1023]} # bottom-left 0.5km by 0.5km region
			#"region":[0,0,4095,4095]}


	test_img = "/data/songtao/metalearning/dataset/boston_task5_small/region12_sat.png"

	test_img = sys.argv[4]
	output_folder = sys.argv[6]
	target_img = sys.argv[5]

	Popen("mkdir -p %s"%output_folder, shell=True).wait()

	# with tf.Session() as sess:
	# 	model = MAML(sess,num_test_updates = 500,inner_lr=0.001)
	# 	model.restoreModel(sys.argv[1])

	# 	Test(example, test_img, model,output_folder = "/data/songtao/metalearning/example/e3/", example_sample=5, color_channel=0, color_intensity = 0.8)


	with tf.Session() as sess:
		model = MAML(sess,num_test_updates = 40,inner_lr=0.001)
		model.restoreModel(sys.argv[1])

		TestIOU(example, test_img,target_img, model,output_folder = output_folder, example_sample=5, color_channel=1, color_intensity = 0.5, threshold = 0.5)




