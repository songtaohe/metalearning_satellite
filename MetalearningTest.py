
import sys
import tensorflow as tf
from PIL import Image
import scipy
import random
import numpy as np 
from time import time 
from subprocess import Popen 
from MetalearningModels import MAMLFirstOrder20191119,MAMLFirstOrder20191119_pyramid



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


def MetaLearnerTrain(model, example, batch_size = 32, image_size = 256):
	traindata = []

	for i in range(len(example['sat'])):
		sat = scipy.ndimage.imread(example['sat'][i]).astype(np.float)/255.0 - 0.5 
		target = scipy.ndimage.imread(example['target'][i]).astype(np.float)/255.0

		if len(np.shape(target)) == 3: 
			target = target[:,:,0]

		x1,y1,x2,y2 = example['region'][i]

		traindata.append([sat[x1:x2,y1:y2], target[x1:x2,y1:y2]])

	example_inputA = np.zeros((batch_size, image_size, image_size ,3))
	example_targetA = np.zeros((batch_size, image_size, image_size, 1))

	# Train the model
	model.meta_lr_val = 0.0001

	it = 101

	IOUs = []
	ts = time()
	s_loss = 0 

	for i in xrange(it):
		loss = 0.0
		#if i > 0:
		for ii in xrange(batch_size):

			ind = random.randint(0, len(traindata)-1)

			dim = np.shape(traindata[ind][0])

			
			x = random.randint(image_size/2, dim[0]-image_size/2*3-1)
			y = random.randint(image_size/2, dim[1]-image_size/2*3-1)

			crop_sat = traindata[ind][0][x-image_size/2:x+image_size/2*3, y-image_size/2:y+image_size/2*3]
			crop_target = traindata[ind][1][x-image_size/2:x+image_size/2*3, y-image_size/2:y+image_size/2*3]

			angle = random.randint(-30,30)
			angle += random.randint(0,3) * 90 


			crop_sat = scipy.ndimage.rotate(crop_sat, angle, reshape=False)
			crop_target = scipy.ndimage.rotate(crop_target, angle, reshape=False)

			example_inputA[ii,:,:,:] = crop_sat[image_size/2:image_size/2*3,image_size/2:image_size/2*3]
			example_targetA[ii,:,:,:] = crop_target[image_size/2:image_size/2*3,image_size/2:image_size/2*3]


		_, loss = model.trainBaselineModel(example_inputA,example_targetA)
		s_loss += loss 

		if i % 10 == 0 and i > 0:
			print(i,s_loss/10.0)
			s_loss = 0.0 

			print("Time left",(time()-ts)*(it-i)/10.0)
			ts = time()

	#print(IOUs)

def MetaLearnerApply(model, sat, output_name, crop_size = 256, stride = 128):
	sat = scipy.ndimage.imread(sat).astype(np.float)/255.0 - 0.5 

	dim = np.shape(sat)

	output = np.zeros((dim[0], dim[1]))
	masks = np.zeros((dim[0],dim[1]))

	for x in range(0, dim[0]-crop_size, stride):
		inputs = np.zeros((dim[1]/stride-1, crop_size, crop_size, 3))
		faketargets = np.zeros((dim[1]/stride-1, crop_size, crop_size, 3))
		
		ii = 0
		for y in range(0, dim[1]-crop_size, stride):
			inputs[ii, :,:,:] = sat[x:x+crop_size, y:y+crop_size]
			ii += 1

		outputs, _= model.runBaselineModel(inputs, faketargets)

		ii = 0
		for y in range(0, dim[1]-crop_size, stride):
			output[x:x+crop_size, y:y+crop_size] += outputs[ii,:,:,0]
			masks[x:x+crop_size, y:y+crop_size] += 1.0
			ii += 1


	output = np.divide(output, masks)

	Image.fromarray((output*255).astype(np.uint8)).save(output_name)
		



if __name__ == "__main__":
	
	#example = {"sat":[sys.argv[2]], 
	#		"target":[sys.argv[3]],
	#		"region":[[2550,450,2550+550,450+550]]}


	#example = {"sat":"data/metalearning_satellite/examples_light_pole/"}

	#test_img = "/data/songtao/metalearning/dataset/boston_task5_small/region12_sat.png"

	#test_img = sys.argv[4]
	
	output_folder = "metalearning_test_output/"
	#target_img = sys.argv[5]

	Popen("mkdir -p %s"%output_folder, shell=True).wait()

	# with tf.Session() as sess:
	# 	model = MAML(sess,num_test_updates = 500,inner_lr=0.001)
	# 	model.restoreModel(sys.argv[1])

	# 	Test(example, test_img, model,output_folder = "/data/songtao/metalearning/example/e3/", example_sample=5, color_channel=0, color_intensity = 0.8)


	with tf.Session() as sess:
		#model = MAML(sess,num_test_updates = 40,inner_lr=0.001)
		model = MAMLFirstOrder20191119_pyramid(sess, num_test_updates = 1,inner_lr=0.001)
		model.restoreModel(sys.argv[1])

		exit() 

		print("Training Metalearning Model")
		MetaLearnerTrain(model, example)
		model.saveModel(output_folder+"model")

		print("Applying Metalearning Model")
		
		

























