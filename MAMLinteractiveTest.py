from MAMLV1_20180429_multidataset import *
#from MAMLV1_20180402_baseline import *

import sys
import tensorflow as tf
from PIL import Image
import scipy
import random
import numpy as np 
from time import time, sleep
from subprocess import Popen
import scipy.ndimage as nd

#isBaseLine = True
isBaseLine = False



def ExtractBoundary(img, threshold=0.5):
	neighbors = nd.convolve((img>threshold).astype(np.int),[[1,1,1],[1,0,1],[1,1,1]],mode='constant',cval=0.0)
	img = np.copy(img)
	img = img / 8 
	img[np.where((neighbors>0)&(neighbors<8))] = 1.0		
	return img 

def TestInt(example_sat,example_target,crop_sat, model,  sat_vis, example_sample = 20, color_channel = 1, color_intensity = 0.5,  output_folder = "", threshold = 0.5):
	
	output_dim = np.shape(crop_sat)

	example_inputA = np.zeros((example_sample, 512,512,3))
	example_targetA = np.zeros((example_sample, 512, 512, 1))

	for i in xrange(example_sample):
		x = random.randint(0, np.shape(example_sat)[0] - 512)
		y = random.randint(0, np.shape(example_sat)[1] - 512)

		example_inputA[i, :,:,:] = example_sat[x:x+512,y:y+512,0:3]
		example_targetA[i,:,:,0] = example_target[x:x+512,y:y+512]

		#Image.fromarray((example_inputA[i, :,:,:]*255).astype(np.uint8)).save(output_folder+"example_%d_sat.png"%i)
		#Image.fromarray((example_targetA[i,:,:,0]*255).astype(np.uint8)).save(output_folder+"example_%d_target.png"%i)

		#example_sat_sample[x:x+512,y:y+512,:] = example_sat_sample[x:x+512,y:y+512,:] + 0.1 

	model.meta_lr_val = 0.0001

	it = 61
	tCNN = time()
	random.seed(123)

	if np.sum(example_target)>10.0:
		for j in xrange(it):
			for i in xrange(example_sample):
				while True:
					x = random.randint(0, np.shape(example_sat)[0] - 512)
					y = random.randint(0, np.shape(example_sat)[1] - 512)

					example_inputA[i, :,:,:] = example_sat[x:x+512,y:y+512,0:3]
					example_targetA[i,:,:,0] = example_target[x:x+512,y:y+512]

					if np.sum(example_targetA[i,:,:,0])>5.0 or i>example_sample*0.6:
						break

					# add roatation and flip 


			_, loss = model.trainBaselineModel(example_inputA,example_targetA)
			if j % 20 == 0:
				print(j,loss)


	test_result = np.zeros((output_dim[0], output_dim[1]))

	batch_size = output_dim[0]/256 * output_dim[1]/256



	test_inputB = np.zeros((batch_size,512,512,3))
	test_outputB= np.zeros((batch_size,512,512,1))
	result= np.zeros((batch_size,512,512,2))

	ind = 0
	for x in xrange(output_dim[0]/256-1):
		for y in xrange(output_dim[1]/256-1):
			test_inputB[ind,:,:,:] = crop_sat[x*256:x*256+512, y*256:y*256+512,:]

			ind += 1

	#print(np.amin(example_targetA),np.amax(example_targetA))

	print(ind)


	for i in xrange(0, ind/16):
		print(i)
		b = min(16,ind - i*16)
		result_, _= model.runBaselineModel(test_inputB[i*16:i*16+b,:,:,:],test_outputB[i*16:i*16+b,:,:,:])
		result[i*16:i*16+b,:,:,:] = result_


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

			test_result[x*256+xl:x*256+xr, y*256+yl:y*256+yr] =  result[ind,xl:xr,yl:yr,0]
			
			ind += 1



	print("CNN time", time() - tCNN)



	# Find top places 
	topScore = np.copy(test_result[:,:])


	top_results = []
	for i in xrange(16):
		max_score = np.amax(topScore)

		if max_score < np.amax(test_result)/2 or max_score < 0.1:
			break

		coords = np.where(topScore>=max_score)
		print(coords)
		max_coord = (coords[0][0],coords[1][0]) 


		sx = max_coord[0] - 128
		sy = max_coord[1] - 128
		if sx <0 :
			sx = 0 
		if sy <0 :
			sy = 0 

		sx = min(sx, output_dim[0]-256-1)
		sy = min(sy, output_dim[1]-256-1)

		topScore[sx:sx+256,sy:sy+256] = 0.0

		print("top",i, sx,sy, max_score)

		top_results.append((sx,sy,max_score))


	rank_output = np.ones((896,512,3)) 

	for i in xrange(16):
		if i >= len(top_results):
			break

		if i < 4:
			y = i % 2 * 256
			x = i / 2 * 256
			rank_output[x:x+256,y:y+256,:] = crop_sat[top_results[i][0]:top_results[i][0]+256, top_results[i][1]:top_results[i][1]+256,:]
			rank_output[x:x+256,y:y+256,1] = rank_output[x:x+256,y:y+256,1] + ExtractBoundary(test_result[top_results[i][0]:top_results[i][0]+256, top_results[i][1]:top_results[i][1]+256])

		else:
			y = ((i-4) % 4) * 128
			x = (i-4)/4 * 128 + 512

			sub_crop1 = np.copy(crop_sat[top_results[i][0]:top_results[i][0]+256, top_results[i][1]:top_results[i][1]+256,:])
			sub_crop1[:,:,1] += ExtractBoundary(test_result[top_results[i][0]:top_results[i][0]+256, top_results[i][1]:top_results[i][1]+256])
			sub_crop1 = np.clip(sub_crop1,0,1.0)
			sub_crop1 = scipy.misc.imresize((sub_crop1*255).astype(np.uint8),(128,128),mode="RGB")

			rank_output[x:x+128,y:y+128,:] = sub_crop1.astype(float)/255.0



	
	rank_output[:,256,:] = 1.0 
	rank_output[256,:,:] = 1.0 
	rank_output[512,:,:] = 1.0
	rank_output[512+128,:,:] = 1.0 
	rank_output[512+256,:,:] = 1.0 
	rank_output[512:,128,:] = 1.0
	rank_output[512:,384,:] = 1.0 


	rank_output = np.clip(rank_output,0,1.0)


	Image.fromarray((rank_output*255).astype(np.uint8)).save(output_folder+"test_output_rank.png")


	output_rgb = np.zeros((output_dim[0],output_dim[1],3), dtype=np.uint8)
	output_rgb[:,:,1] = (test_result*255).astype(np.uint8)
	#output_rgb[:,:,3] = (test_result*255).astype(np.uint8)

	output_rgb = scipy.misc.imresize(output_rgb, (800,800),mode="RGB")
	output_rgb = output_rgb.astype(int)
	output_rgb = output_rgb + sat_img_vis

	output_rgb = np.clip(output_rgb, 0,255) 

	for i in xrange(16):
		if i >= len(top_results):
			break

		top_results[i]= (top_results[i][0] * 800 / output_dim[0], top_results[i][1] * 800 / output_dim[1])
		
		dim = 256 * 800/output_dim[0]

		output_rgb[top_results[i][0]:top_results[i][0]+dim,top_results[i][1],:] = 255
		output_rgb[top_results[i][0]:top_results[i][0]+dim,top_results[i][1]+dim,:] = 255

		output_rgb[top_results[i][0],top_results[i][1]:top_results[i][1]+dim,:] = 255
		output_rgb[top_results[i][0]+dim,top_results[i][1]:top_results[i][1]+dim,:] = 255



	Image.fromarray(output_rgb.astype(np.uint8)).save(output_folder+"test_output_BW.png")

	#Image.fromarray((crop_sat*255).astype(np.uint8)).save(output_folder+"test_output_test.png")

	print(np.amin(test_result),np.amax(test_result))

	# crop_sat[:,:,color_channel] = np.clip(crop_sat[:,:,color_channel] + test_result[:,:]*color_intensity,0.0,1.0)

	# Image.fromarray((crop_sat*255).astype(np.uint8)).save(output_folder+"test_output_soft.png")
	# #test_result = test_result*test_result
	# Image.fromarray((test_result*255).astype(np.uint8)).save(output_folder+"test_output_BW.png")

	

#dataset/boston_task5_small/region12_sat.png


if __name__ == "__main__":
# load model
# test on examples 
	random.seed(321)

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

	# example = {"sat":sys.argv[2], 
	# 		"target":sys.argv[3],
	# 		#"region":[2048,2048,4095-1024,4095-1024]}
	# 		"region":[1024,1024,2048,2048]}


	# test_img = "/data/songtao/metalearning/dataset/boston_task5_small/region12_sat.png"

	# test_img = sys.argv[4]
	# target_img = sys.argv[5]
	# output_folder = sys.argv[6]
	

	# Popen("mkdir -p %s"%output_folder, shell=True).wait()

	# with tf.Session() as sess:
	# 	model = MAML(sess,num_test_updates = 500,inner_lr=0.001)
	# 	model.restoreModel(sys.argv[1])

	# 	Test(example, test_img, model,output_folder = "/data/songtao/metalearning/example/e3/", example_sample=5, color_channel=0, color_intensity = 0.8)

	# train_sat_img = sys.argv[2]
	# train_target_img = sys.argv[3]
	# test_sat_img = sys.argv[4]
	# output_folder = sys.argv[5]

	train_sat_img = "/data/songtao/upload/train_sat.png"
	train_target_img = "/data/songtao/upload/target.png"
	test_sat_img = "/data/songtao/upload/test_sat_img.png"
	output_folder = "/data/songtao/upload/"


	working_folder = "/data/songtao/metalearning/code/upload/"
	cmd_text = working_folder+"cmd.txt"
	input_sat_file = working_folder+"input_sat.png"
	input_sat_file = working_folder+"input_sat_vis.png"
	example_mask_file = working_folder+"example_mask.png"
	example_sat_file = working_folder+"example_sat.png"
	output_folder = working_folder

	#test_sat = scipy.ndimage.imread(test_sat_img).astype(np.float)/255.0  
	#crop_sat = test_sat[1024:3072,1024:3072,:]

	#example_sat = scipy.ndimage.imread(train_sat_img).astype(np.float)/255.0  
	#example_target = scipy.ndimage.imread(train_target_img).astype(np.float)/255.0  



	sat_path = None 
	region_cur = None 

	reload_sat = True
	reload_example_region = True

	pre_example_target = None 
	with tf.Session() as sess:
		model = MAML(sess,num_test_updates = 40,inner_lr=0.001)
		#model.restoreModel(sys.argv[1])

		while True:
			model.restoreModel(sys.argv[1])
			print("Try Load CMD")
			try:
				with open(cmd_text,"r") as fin:
					line = fin.readlines()[0]
					if line.split(';')[1] != sat_path:
						sat_path = line.split(';')[1].split("\n")[0]
						reload_sat = True 
						reload_example_region = True 

						if len(sat_path.split(',')) > 2:
							Popen("cp %s %s" % (sat_path.split(',')[0], input_sat_file), shell=True).wait()
						else:
							Popen("cp %s %s" % (sat_path, input_sat_file), shell=True).wait()




					if line.split(';')[0] != region_cur:
						region_cur = line.split(';')[0]
						reload_example_region = True 


			except:
				sleep(0.1)
				continue

			if reload_sat == True:
				try: 
					sat_img_ = scipy.ndimage.imread(input_sat_file).astype(np.float)/255.0  
					#print(sat_path.split(','))
					if len(sat_path.split(',')) > 2:
						sat_img = sat_img_[int(sat_path.split(',')[1]):int(sat_path.split(',')[2]), int(sat_path.split(',')[3]):int(sat_path.split(',')[4]),:]
					else:
						sat_img = sat_img_
					sat_img_vis = scipy.misc.imresize((sat_img*255).astype(np.uint8),(800,800),mode="RGB")  

					

				except:
					print("reload sat")
					sleep(0.1)
					continue

				reload_sat = False

			if reload_example_region == True:

				x = float(region_cur.split(',')[0])*np.shape(sat_img)[0]
				y = float(region_cur.split(',')[1])*np.shape(sat_img)[1]

				x = int(x)
				y = int(y)


				if len(region_cur.split(','))>2:
					dimx = int(float(region_cur.split(',')[2])*np.shape(sat_img)[0])
					dimy = int(float(region_cur.split(',')[3])*np.shape(sat_img)[1])
				else:
					dimx = 512
					dimy = 512

				dimx = max(512,dimx)
				dimy = max(512,dimy)

				print(x,y,dimx,dimy)

				x = min(np.shape(sat_img)[0]-dimx,x)
				y = min(np.shape(sat_img)[1]-dimy,y)

				example_sat = np.copy(sat_img[x:x+dimx,y:y+dimy,:])

				Image.fromarray((example_sat*255).astype(np.uint8)).save(example_sat_file)

				reload_example_region = False 

			try: 
				example_target = scipy.ndimage.imread(example_mask_file)
				example_target = example_target[:,:,1]
				example_target = scipy.misc.imresize(example_target, (dimx,dimy))
				example_target = example_target.astype(np.float)/255.0  
				
				example_target[np.where(example_target>0.1)] = 1.0


			except:
				print("reload example",dimx,dimy)
				sleep(0.1)
				continue

			



			ts0 = time()
			

			if np.sum(example_target) < 1.0 :
				Image.fromarray(sat_img_vis).save(output_folder+"test_output_BW.png")
				sleep(1)
				continue

			if pre_example_target is not None :
				if abs(np.sum(pre_example_target) - np.sum(example_target))<1.0:
					sleep(1)
					continue


			pre_example_target = example_target

			TestInt(example_sat,example_target,sat_img, model, sat_img_vis.astype(float)/255.0,output_folder = output_folder, example_sample=5, color_channel=1, color_intensity = 0.5, threshold = 0.5)
			print(time() - ts0)



