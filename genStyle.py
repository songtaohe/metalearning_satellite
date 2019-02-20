import sys, getopt, os
from subprocess import Popen
import math
import numpy as np 
import scipy.misc
import scipy
import scipy.ndimage
from PIL import Image

from time import time

import inputFilter



if __name__ == "__main__":


	file_name = sys.argv[1]
	img = scipy.ndimage.imread(file_name)

	for i in xrange(0,10):
		print(file_name.replace('sat.png',"sat%d.png"%i),i)
		new_img = inputFilter.applyFilter(img,i)

		try:
			Image.fromarray(new_img).save(file_name.replace('sat.png',"sat%d.png"%i))
		except:
			print("Error!!!")
			print(np.shape(new_img))


	exit()


	tasks = []

	cityname = os.listdir(sys.argv[1])
	#output_folder = sys.argv[2]
	for name in cityname:
		# if name == "dc" or name == "chicago" or name == "houston":
		# 	continue 
		# else:
		# 	pass 


		folder = sys.argv[1] + "/" + name + "/"

		sat_image = np.zeros((1024*6,1024*6,3))
		bd = np.zeros((1024,1024,3))

		bd[:,0:5,:] = 1
		bd[:,1018:1024,:] = 1
		bd[0:5,:,:] = 1
		bd[1018:1024,:,:] = 1

		
		for i in xrange(6):
			for j in xrange(6):
				try:
					ind = j*6 + i 
					print(name, ind)

					sat = scipy.ndimage.imread(folder +"region%d_sat.png"%ind)
					sat = scipy.misc.imresize(sat, (1024,1024),mode="RGB")

					sat_image[(5-j)*1024:(5-j)*1024+1024, i*1024:i*1024+1024,:] = sat


					road = scipy.ndimage.imread(folder +"region%d_t2.png"%ind)
					building = scipy.ndimage.imread(folder +"region%d_t4.png"%ind)

					road = scipy.misc.imresize(road, (1024,1024))
					building = scipy.misc.imresize(building, (1024,1024))



					sat_image[(5-j)*1024:(5-j)*1024+1024, i*1024:i*1024+1024,1] += building/3
					sat_image[(5-j)*1024:(5-j)*1024+1024, i*1024:i*1024+1024,2] += road/3
					sat_image[(5-j)*1024:(5-j)*1024+1024, i*1024:i*1024+1024,:] -= bd * 255

				except:
					print("Error")
					pass


		sat_image = np.clip(sat_image,0,255)

		Image.fromarray(sat_image.astype(np.uint8)).save(folder+name+"_sketch.png")














	# 	with open(sys.argv[1] + "/" + name + "/boundingbox.txt", "r") as fin:
	# 		items = fin.readlines()[0].split(' ')

	# 		#print(name, items)

	# 		region = [float(x) for x in items]
			
	# 		print(name, region)

	# 		tasks.append([region, sys.argv[1] + "/" + name+"/", output_folder + "/"+ name +"/", sys.argv[1] + "/" + name + "/cityname.osm.pbf"])

	# for task in tasks:
	# 	Popen("python MetaTaskGenBatch.py %f %f %s %s %s "% (task[0][0], task[0][1], task[1], task[2], task[3]),shell=True).wait()



	


