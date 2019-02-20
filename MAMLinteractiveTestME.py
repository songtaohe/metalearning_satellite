from MAMLV1_20180429_multidataset import *
#from SimpleCNN import *
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
import json
from PIL import ImageFont
from PIL import ImageDraw
import math


#isBaseLine = True
isBaseLine = False

scale = 2
drawBlock = False


def ExtractBoundary(img, threshold=0.5, scale=0.125):
	neighbors = nd.convolve((img>threshold).astype(np.int),[[1,1,1],[1,0,1],[1,1,1]],mode='constant',cval=0.0)
	img = np.copy(img)
	img = img * scale
	img[np.where((neighbors>0)&(neighbors<8))] = 1.0		
	return img 



def TestVideo(model):
	max_frame = 6000

	# path_in = "/data/songtao/metalearning/dataset_video/car_drone/frames_input/"
	# path_out = "/data/songtao/metalearning/dataset_video/car_drone/frames_output/"
	# path_out2 = "/data/songtao/metalearning/dataset_video/car_drone/frames_output2/"
	# path_out3 = "/data/songtao/metalearning/dataset_video/car_drone/frames_output3/"

	# path_in = "/data/songtao/metalearning/dataset_video/dubai/frames_input/"
	# path_out = "/data/songtao/metalearning/dataset_video/dubai/frames_output/"
	# path_out2 = "/data/songtao/metalearning/dataset_video/dubai/frames_output2/"
	# path_out3 = "/data/songtao/metalearning/dataset_video/dubai/frames_output3/"

	# path_in = "/data/songtao/metalearning/dataset_video/london/frames_input/"
	# path_out = "/data/songtao/metalearning/dataset_video/london/frames_output/"
	# path_out2 = "/data/songtao/metalearning/dataset_video/london/frames_output2/"
	# path_out3 = "/data/songtao/metalearning/dataset_video/london/frames_output3/"

	path_in = "/data/songtao/metalearning/dataset_video/la_people/frames_input/"
	path_out = "/data/songtao/metalearning/dataset_video/la_people/frames_output/"
	path_out2 = "/data/songtao/metalearning/dataset_video/la_people/frames_output2/"
	path_out3 = "/data/songtao/metalearning/dataset_video/la_people/frames_output3/"




	zeros_pad = None 
	input_ = None 
	output_ = None 


	test_inputB = None 
	test_outputB = None 
	result = None
	input_frame_pad = None 
	result_pad = None 

	isInit = False

	for frame_i in xrange(max_frame):
		input_frame = scipy.ndimage.imread(path_in + "%04d.png"%(frame_i+1)).astype(float)/255.0
		dimx = np.shape(input_frame)[0]
		dimy = np.shape(input_frame)[1]
		

		if zeros_pad is None :
			zeros_pad = np.zeros((1,dimx,dimy,1))
		if input_ is None:
			input_ = np.zeros((1,dimx,dimy,3))
			output_ = np.zeros((dimx,dimy,3))

		if isInit==False:
			output_dim = ((dimx/256+1)*256, (dimy/256+1)*256)
			batch_size = (dimx/256+1) * (dimy/256+1)
			input_frame_pad = np.zeros(((dimx/256+1)*256, (dimy/256+1)*256, 3))
			result_pad = np.zeros(((dimx/256+1)*256, (dimy/256+1)*256))

			test_inputB = np.zeros((batch_size,512,512,3))
			test_outputB= np.zeros((batch_size,512,512,1))
			result= np.zeros((batch_size,512,512,2))

			isInit = True


		input_frame_pad[0:dimx,0:dimy,:] = input_frame


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
			result_, _= model.runBaselineModel(test_inputB[i*32:i*32+b,:,:,:],test_outputB[i*32:i*32+b,:,:,:])
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

		# input_[0,:,:,:] = input_frame



		# result,_,_ = model.runBaselineModel(input_,zeros_pad)

		#input_frame[:,:,0] += ExtractBoundary(result_pad[0:dimx,0:dimy],threshold=0.3,scale=0.125)
		
		input_frame[:,:,1] += ExtractBoundary(result_pad[0:dimx,0:dimy],threshold=0.5,scale=0.3)
		input_frame[:,:,0] -= result_pad[0:dimx,0:dimy] * 0.5
		input_frame[:,:,2] -= result_pad[0:dimx,0:dimy] * 0.5 


		input_frame = np.clip(input_frame,0,1)

		Image.fromarray((input_frame*255).astype(np.uint8)).save(path_out2 + "%04d.png"%(frame_i+1))

		Image.fromarray((result_pad[0:dimx,0:dimy]*255).astype(np.uint8)).save(path_out3 + "%04d.png"%(frame_i+1))



		if frame_i %10 == 0:
			print(frame_i,max_frame)



def TestWholeCity(model, path):

	t_total = time()
	t_cnn = 0


	items = path.split("/")
	path = "/"
	for item in items[:-1]:
		path += item + "/"

	

	stylystr = items[-1].split('.')[0].split('_')[1]
		
	print(stylystr)




	print(path)
	Popen("mkdir -p upload/whole_city_test_%s/"%items[-2], shell=True).wait()
	Popen("rm upload/whole_city_test_%s/*"%items[-2], shell=True).wait()

	output_folder = "upload/whole_city_test_%s/"%items[-2]


	sat_vis = np.zeros((36,1024,1024,3),dtype=np.uint8)


	all_top_result = []

	for rid in xrange(36):

		progress = np.ones((2,36), dtype=np.uint8)*255
		progress[:,0:rid] = 0 

		Image.fromarray(progress).save("upload/example_overview.png")

		print(rid)
		crop_sat = scipy.ndimage.imread(path+"region%d_%s.png"%(rid,stylystr))

		output_dim = np.shape(crop_sat)

		crop_sat = scipy.misc.imresize(crop_sat,(output_dim[0]/scale, output_dim[1]/scale),mode="RGB")

		crop_sat = crop_sat.astype(float)/255.0
		
		output_dim = np.shape(crop_sat)
		test_result = np.zeros((output_dim[0], output_dim[1]))
		batch_size = (output_dim[0]/256) * (output_dim[1]/256)
		test_inputB = np.zeros((batch_size,512,512,3))
		test_outputB= np.zeros((batch_size,512,512,1))
		result= np.zeros((batch_size,512,512,2))

		ind = 0
		for x in xrange(output_dim[0]/256-1):
			for y in xrange(output_dim[1]/256-1):
				test_inputB[ind,:,:,:] = crop_sat[x*256:x*256+512, y*256:y*256+512,:]

				ind += 1

		#print(np.amin(example_targetA),np.amax(example_targetA))

		#print(ind)

		t0 = time()
		for i in xrange(0, ind/16+1):
			#print(i)
			b = min(16,ind - i*16)
			if b == 0:
				continue
			result_, _= model.runBaselineModel(test_inputB[i*16:i*16+b,:,:,:],test_outputB[i*16:i*16+b,:,:,:])
			result[i*16:i*16+b,:,:,:] = result_

		t_cnn += time()-t0


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


		topScore = np.copy(test_result)

		top_results = []
		for i in xrange(256):
			max_score = np.amax(topScore)

			if max_score < np.amax(test_result)/2 or max_score < 0.1:
				break

			coords = np.where(topScore>=max_score)
			#print(coords)
			max_coord = (coords[0][0],coords[1][0]) 


			sx = max_coord[0] - 128
			sy = max_coord[1] - 128
			if sx <0 :
				sx = 0 
			if sy <0 :
				sy = 0 

			sx = min(sx, output_dim[0]-256)
			sy = min(sy, output_dim[1]-256)

			topScore[sx:sx+256,sy:sy+256] = 0.0

			#print("top",i, sx,sy, max_score)


			crop = np.copy(crop_sat[sx:sx+256, sy:sy+256,:])

			crop[:,:,1] = crop[:,:,1] + ExtractBoundary(test_result[sx:sx+256, sy:sy+256])
			crop = np.clip(crop,0,1)

			top_results.append((crop,max_score,rid, sx,sy))


		test_result = np.power(test_result,0.2)

		Image.fromarray((test_result*255).astype(np.uint8)).save(output_folder+"region%d_output.png"%rid)

		crop_sat[:,:,1] = np.clip(crop_sat[:,:,1]+test_result*0.5,0,1)
		sat_vis[rid,:,:,:] = scipy.misc.imresize((crop_sat*255).astype(np.uint8),(1024,1024),mode="RGB")


		test_result = (test_result*255).astype(np.uint8)

		for iix in xrange(8):
			for iiy in xrange(8):
				test_result[output_dim[0]/8*iix:output_dim[0]/8*(iix+1),output_dim[1]/8*iiy:output_dim[1]/8*(iiy+1)] = np.sum(test_result[output_dim[0]/8*iix:output_dim[0]/8*(iix+1),output_dim[1]/8*iiy:output_dim[1]/8*(iiy+1)])/(output_dim[0]*output_dim[1]/64)


		

		Image.fromarray(test_result).save(output_folder+"region%d_output_priority.png"%rid)


		#Image.fromarray(sat_vis[rid,:,:,:]).save(output_folder+"region%d.png"%rid)


		print(len(top_results))

		all_top_result  = all_top_result  + top_results


		all_top_result = sorted(all_top_result, key=lambda item: item[1],reverse=True)


		if len(all_top_result)>256:
			all_top_result = all_top_result[0:256]


		print(len(all_top_result))



	for i in xrange(len(all_top_result)):
		print(all_top_result[i][1])
		if all_top_result[i][1] < all_top_result[0][1]/2 :
			break


		all_top_result[i][0][0,:,:] = 255
		all_top_result[i][0][255,:,:] = 255
		all_top_result[i][0][:,0,:] = 255
		all_top_result[i][0][:,255,:] = 255


		Img = Image.fromarray((all_top_result[i][0]*255).astype(np.uint8))
		draw = ImageDraw.Draw(Img)
		font = ImageFont.load_default()

		draw.text((10, 19),"Rank %d, Score %.3f, Region ID %d"%(i+1,all_top_result[i][1],all_top_result[i][2]),(255,255,255),font=font)

		Img.save(output_folder+"rank%d.png"%i)



		rid = all_top_result[i][2]
		sx = all_top_result[i][3]/4
		sy = all_top_result[i][4]/4 

		if drawBlock :

			sat_vis[rid,sx:sx+2,sy:sy+63,:] = 255
			sat_vis[rid,sx+61:sx+63,sy:sy+63,:] = 255
			sat_vis[rid,sx:sx+63,sy:sy+2,:] = 255
			sat_vis[rid,sx:sx+63,sy+61:sy+63,:] = 255




	for rid in xrange(36):
		Image.fromarray(sat_vis[rid,:,:,:]).save(output_folder+"region%d.png"%rid)




	print("Time of running CNN model", t_cnn)
	print("Total time", time()-t_total)
	print("Ratio", t_cnn/(time()-t_total))







def TestInt(example_regions,crop_sat, model,  sat_vis, example_sample = 20, color_channel = 1, color_intensity = 0.5,  output_folder = "", threshold = 0.5, test_whole_city=None, iteration=10, lr = 0.0001):
	



	output_dim = np.shape(crop_sat)

	example_inputA = np.zeros((example_sample, 512,512,3))
	example_targetA = np.zeros((example_sample, 512, 512, 1))

	# for i in xrange(example_sample):
	# 	x = random.randint(0, np.shape(example_sat)[0] - 512)
	# 	y = random.randint(0, np.shape(example_sat)[1] - 512)

	# 	example_inputA[i, :,:,:] = example_sat[x:x+512,y:y+512,0:3]
	# 	example_targetA[i,:,:,0] = example_target[x:x+512,y:y+512]

	# 	#Image.fromarray((example_inputA[i, :,:,:]*255).astype(np.uint8)).save(output_folder+"example_%d_sat.png"%i)
		#Image.fromarray((example_targetA[i,:,:,0]*255).astype(np.uint8)).save(output_folder+"example_%d_target.png"%i)

		#example_sat_sample[x:x+512,y:y+512,:] = example_sat_sample[x:x+512,y:y+512,:] + 0.1 

	#model.meta_lr_val = 0.0001
	model.meta_lr_val = lr
	it = iteration+1
	tCNN = time()
	random.seed(123)

	
	for j in xrange(it):
		for i in xrange(example_sample):
			cc = 0 
			while True:

				rid = random.randint(0,len(example_regions)-1)

				if example_regions[rid][0] is None or example_regions[rid][1] is None:
					continue

				example_sat = example_regions[rid][0]
				example_target = example_regions[rid][1]

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
				# add roatation and flip 


		_, loss = model.trainBaselineModel(example_inputA,example_targetA)
		if j % 20 == 0:
			print(j,loss)


	test_result = np.zeros((output_dim[0], output_dim[1]))

	batch_size = (output_dim[0]/256) * (output_dim[1]/256)



	if test_whole_city is not None:
		TestWholeCity(model, test_whole_city)

	#TestVideo(model)
	#exit()

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


	for i in xrange(0, ind/32+1):
		print(i)
		b = min(32,ind - i*32)
		if b == 0:
			continue
		result_, _= model.runBaselineModel(test_inputB[i*32:i*32+b,:,:,:],test_outputB[i*32:i*32+b,:,:,:])
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

			test_result[x*256+xl:x*256+xr, y*256+yl:y*256+yr] =  result[ind,xl:xr,yl:yr,0]
			
			ind += 1



	print("CNN time", time() - tCNN)



	# Find top places 
	topScore = np.copy(test_result[:,:])


	top_results = []
	for i in xrange(24):
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


	rank_output = np.ones((896,768,3)) 

	for i in xrange(24):
		if i >= len(top_results):
			break

		if i < 6:
			y = i % 3 * 256
			x = i / 3 * 256
			rank_output[x:x+256,y:y+256,:] = crop_sat[top_results[i][0]:top_results[i][0]+256, top_results[i][1]:top_results[i][1]+256,:]
			rank_output[x:x+256,y:y+256,1] = rank_output[x:x+256,y:y+256,1] + ExtractBoundary(test_result[top_results[i][0]:top_results[i][0]+256, top_results[i][1]:top_results[i][1]+256])
			#rank_output[x:x+256,y:y+256,0] = rank_output[x:x+256,y:y+256,0] + ExtractBoundary(test_result[top_results[i][0]:top_results[i][0]+256, top_results[i][1]:top_results[i][1]+256])

		else:
			y = ((i-6) % 6) * 128
			x = (i-6)/6 * 128 + 512

			sub_crop1 = np.copy(crop_sat[top_results[i][0]:top_results[i][0]+256, top_results[i][1]:top_results[i][1]+256,:])
			#sub_crop1[:,:,0] += ExtractBoundary(test_result[top_results[i][0]:top_results[i][0]+256, top_results[i][1]:top_results[i][1]+256])
			sub_crop1[:,:,1] += ExtractBoundary(test_result[top_results[i][0]:top_results[i][0]+256, top_results[i][1]:top_results[i][1]+256])
			
			sub_crop1 = np.clip(sub_crop1,0,1.0)
			sub_crop1 = scipy.misc.imresize((sub_crop1*255).astype(np.uint8),(128,128),mode="RGB")

			rank_output[x:x+128,y:y+128,:] = sub_crop1.astype(float)/255.0



	
	rank_output[:,256,:] = 1.0 
	rank_output[:,512,:] = 1.0 

	rank_output[256,:,:] = 1.0 
	rank_output[512,:,:] = 1.0

	rank_output[512+128,:,:] = 1.0 
	rank_output[512+256,:,:] = 1.0 

	rank_output[512:,128,:] = 1.0
	rank_output[512:,384,:] = 1.0 
	rank_output[512:,384+256,:] = 1.0 


	rank_output = np.clip(rank_output,0,1.0)


	Image.fromarray((rank_output*255).astype(np.uint8)).save(output_folder+"test_output_rank.png")


	output_rgb = np.zeros((output_dim[0],output_dim[1],3), dtype=np.uint8)


	output_rgb[:,:,1] = (test_result*255).astype(np.uint8)
	#output_rgb[:,:,0] = (test_result*255).astype(np.uint8)

	#output_rgb[:,:,3] = (test_result*255).astype(np.uint8)

	output_rgb = scipy.misc.imresize(output_rgb, (800,800),mode="RGB")
	output_rgb = output_rgb.astype(int)
	output_rgb = output_rgb + sat_img_vis

	output_rgb = np.clip(output_rgb, 0,255) 

	for i in xrange(24):
		if i >= len(top_results):
			break

		top_results[i]= (top_results[i][0] * 800 / output_dim[0], top_results[i][1] * 800 / output_dim[1])
		
		dimx = 256 * 800/output_dim[0]
		dimy = 256 * 800/output_dim[1]

		if drawBlock :

			output_rgb[top_results[i][0]:top_results[i][0]+dimx,top_results[i][1],:] = 255
			output_rgb[top_results[i][0]:top_results[i][0]+dimx,top_results[i][1]+dimy,:] = 255

			output_rgb[top_results[i][0],top_results[i][1]:top_results[i][1]+dimy,:] = 255
			output_rgb[top_results[i][0]+dimx,top_results[i][1]:top_results[i][1]+dimy,:] = 255



	Image.fromarray(output_rgb.astype(np.uint8)).save(output_folder+"test_output_BW.png")


	#Image.fromarray((crop_sat*255).astype(np.uint8)).save(output_folder+"test_output_test.png")

	print(np.amin(test_result),np.amax(test_result))


	output_rgb[:,:,0] = 0
	output_rgb[:,:,1] = scipy.misc.imresize((test_result*255).astype(np.uint8),(800,800))
	output_rgb[:,:,2] = 0


	output_rgb = output_rgb.astype(np.uint8)
	output_rgb = scipy.misc.imresize(output_rgb,(512,512),mode="RGB")

	Image.fromarray(output_rgb).save(output_folder+"test_output_mask.png")


	crop_sat_ = np.copy(crop_sat)

	crop_sat_[:,:,color_channel] = np.clip(crop_sat[:,:,color_channel] + test_result[:,:]*color_intensity,0.0,1.0)

	Image.fromarray((crop_sat_*255).astype(np.uint8)).save(output_folder+"test_output_soft_highres.png")
	# #test_result = test_result*test_result
	Image.fromarray((test_result*255).astype(np.uint8)).save(output_folder+"test_output_BW_highres.png")

	

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
	json_file = working_folder+"state.json"
	input_sat_file = working_folder+"input_sat.png"
	input_sat_file = working_folder+"input_sat_vis.png"
	example_mask_file = working_folder+"example_mask.png"
	example_sat_file = working_folder+"example_sat.png"
	example_overview = working_folder+"example_overview.png"

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

	cur_state = {}
	cur_state["test_region"] = ""
	cur_state["example_regions"] = []
	cur_state["test_whole_city"] = 0
	cur_state["test_iterations"] = 20
	cur_state["test_LR"] = 0.0001

	example_regions = []

	state_ready = False

	example_region_overview = np.ones((256, 1024,3),dtype=np.uint8)*255

	sat_img = None 
	with tf.Session() as sess:
		model = MAML(sess,num_test_updates = 40,inner_lr=0.001)
		#model.restoreModel(sys.argv[1])

		while True:
			change = False
			if len(sys.argv)>1:
				model.restoreModel(sys.argv[1])

			print("load state")

			try:
				with open(json_file,"r") as fin:
					new_state = json.load(fin)

			except:
				print("load json failed")
				sleep(0.1)
				continue





			if new_state["test_region"] != cur_state["test_region"]:
				#cur_state["test_region"] = new_state["test_region"]
				if new_state["test_region"] == "":
					cur_state["test_region"] = ""
					state_ready = False
					sleep(0.1)
					continue
				else:
					try: 
						sat_path = new_state["test_region"].split(";")[0]
						sat_path_file = sat_path.split(',')[0]

						sat_img_ = scipy.ndimage.imread(sat_path_file) 

						dimx = np.shape(sat_img_)[0]
						dimy = np.shape(sat_img_)[1]

						sat_img_ = scipy.misc.imresize(sat_img_,(dimx/scale, dimy/scale),mode="RGB")
						sat_img_ = sat_img_.astype(np.float)/255.0
						dimx = np.shape(sat_img_)[0]
						dimy = np.shape(sat_img_)[1]


						sat_img__ = np.zeros((int(math.ceil(dimx/256.0))*256,int(math.ceil(dimy/256.0))*256,3))

						sat_img__[0:dimx, 0:dimy, :] = sat_img_
						sat_img_ = sat_img__



						#print(sat_path.split(','))
						if len(sat_path.split(',')) > 2:
							sat_img = sat_img_[int(sat_path.split(',')[1]):int(sat_path.split(',')[2]), int(sat_path.split(',')[3]):int(sat_path.split(',')[4]),:]
						else:
							sat_img = sat_img_
						sat_img_vis = scipy.misc.imresize((sat_img*255).astype(np.uint8),(800,800),mode="RGB")  

					except:
						print("reload sat", new_state["test_region"])
						sleep(0.1)
						continue

					cur_state["test_region"] = new_state["test_region"]
					change=True


			for i in xrange(max(len(new_state["example_regions"]),len(cur_state["example_regions"]))):
				if i < len(cur_state["example_regions"]) and i < len(new_state["example_regions"]):
					if new_state["example_regions"][i] == cur_state["example_regions"][i]:
						continue 
						# update it 
				
				if i < len(new_state["example_regions"]):
					try:
						region_tmp_sat = scipy.misc.imread(new_state["example_regions"][i][1])
						region_cur = new_state["example_regions"][i][2]

						x = float(region_cur.split(',')[0])*np.shape(region_tmp_sat)[0]
						y = float(region_cur.split(',')[1])*np.shape(region_tmp_sat)[1]

						x = int(x)
						y = int(y)


						if len(region_cur.split(','))>2:
							dimx = int(float(region_cur.split(',')[2])*np.shape(region_tmp_sat)[0])
							dimy = int(float(region_cur.split(',')[3])*np.shape(region_tmp_sat)[1])
						else:
							dimx = 512
							dimy = 512

						dimx = max(512,dimx)
						dimy = max(512,dimy)

						print(x,y,dimx,dimy)

						x = min(np.shape(region_tmp_sat)[0]-dimx,x)
						y = min(np.shape(region_tmp_sat)[1]-dimy,y)

						example_sat = np.copy(region_tmp_sat[x:x+dimx,y:y+dimy,:])

						

						Image.fromarray(example_sat).save(working_folder+"example_sat%d.png"%i)

						loc_x = i / 8 * 128
						loc_y = i % 8 * 128

						example_region_overview[loc_x:loc_x+128,loc_y:loc_y+128,:] = scipy.misc.imresize(example_sat,(128,128),mode="RGB")

					except:
						print("Load region %s error"%new_state["example_regions"][i])
						sleep(0.1)
						continue

					if i >= len(cur_state["example_regions"]):
						cur_state["example_regions"].append(new_state["example_regions"][i])
						example_regions.append([example_sat.astype(float)/255.0,None])
					else:
						cur_state["example_regions"][i] = new_state["example_regions"][i]
						example_regions[i] = [example_sat.astype(float)/255.0,None]



				else:
					if i< len(cur_state["example_regions"]):
						loc_x = i / 8 * 128
						loc_y = i % 8 * 128
						cur_state["example_regions"][i][0] = "invalid"
						example_regions[i] = [None,None]
						example_region_overview[loc_x:loc_x+128,loc_y:loc_y+128,:] = 255
				

					
			# Output the overview 

			#Image.fromarray(example_region_overview).save(example_overview)

			example_region_overview_label = np.copy(example_region_overview)

			print("Try Load CMD")
			


			valid_list = []
			


			for i in xrange(len(example_regions)):
				if example_regions[i][0] is None:
					continue

				try: 
					dimx = np.shape(example_regions[i][0])[0]
					dimy = np.shape(example_regions[i][0])[1]

					example_target_ = scipy.ndimage.imread(working_folder+"example_mask%d.png" % i)
					example_target_ = example_target_[:,:,1]
					example_target = scipy.misc.imresize(example_target_, (dimx,dimy))
					example_target = example_target.astype(np.float)/255.0  
					
					example_target[np.where(example_target>0.1)] = 1.0

					#example_regions[i][1] = example_target
				except:
					print("reload example failed",working_folder+"example_mask%d.png" % i)
					sleep(0.1)
					continue


				if example_regions[i][1] is not None:
					if abs(np.sum(example_regions[i][1]) - np.sum(example_target))>1.0:
						change=True
				else:
					change=True

				example_regions[i][1] = example_target

				loc_x = i / 8 * 128
				loc_y = i % 8 * 128

				example_region_overview_label[loc_x:loc_x+128,loc_y:loc_y+128,1] = np.clip(example_region_overview_label[loc_x:loc_x+128,loc_y:loc_y+128,1]+scipy.misc.imresize(example_target_, (128,128))/2,0,255)

				

				# if np.sum(example_target) > 1.0:
				# 	change=True

				valid_list.append(i)

			example_region_overview_label[128,:,:] = 128
			for i in xrange(1,8):
				example_region_overview_label[:,128*i,:] = 128

			Image.fromarray(example_region_overview_label).save(example_overview)

			print("Valid List", valid_list)

			if len(valid_list) == 0 :
				sleep(1.0)
				continue 



			if cur_state["test_iterations"] != new_state["test_iterations"]:
				cur_state["test_iterations"] = new_state["test_iterations"]
				change = True 

			if cur_state["test_LR"] != new_state["test_LR"]:
				cur_state["test_LR"] = new_state["test_LR"]
				change = True 


			if change==False and cur_state["test_whole_city"] == new_state["test_whole_city"]:
				sleep(1.0)
				continue

			ts0 = time()
			
			# if np.sum(example_target) < 1.0 :
			# 	Image.fromarray(sat_img_vis).save(output_folder+"test_output_BW.png")
			# 	sleep(1)
			# 	continue

			# if pre_example_target is not None :
			# 	if abs(np.sum(pre_example_target) - np.sum(example_target))<1.0:
			# 		sleep(1)
			# 		continue
			#pre_example_target = example_target
			if sat_img is None:
				sleep(1.0)
				continue


			if cur_state["test_whole_city"] != new_state["test_whole_city"]:
				cur_state["test_whole_city"]=new_state["test_whole_city"]
				TestInt(example_regions,sat_img, model, sat_img_vis.astype(float)/255.0,output_folder = output_folder, example_sample=5, color_channel=0, color_intensity = 0.9, threshold = 0.5, test_whole_city=cur_state["test_region"].split(";")[0],iteration = cur_state["test_iterations"], lr=cur_state["test_LR"])
			else:
				TestInt(example_regions,sat_img, model, sat_img_vis.astype(float)/255.0,output_folder = output_folder, example_sample=5, color_channel=0, color_intensity = 0.9, threshold = 0.5, iteration = cur_state["test_iterations"],lr=cur_state["test_LR"])


			print(time() - ts0)



