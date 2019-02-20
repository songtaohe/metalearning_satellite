import sys, getopt, os
from subprocess import Popen
import math
from time import time,sleep
import scipy.ndimage
import numpy as np 
from  PIL import Image

copy_cmd = "scp -i ~/.ssh/songtao_cmt %s ubuntu@ec2-34-211-12-193.us-west-2.compute.amazonaws.com:/data/songtao/upload/target.png"

if __name__ == "__main__":
	folder = sys.argv[1]

	while True:
		files = os.listdir(folder)

		if len(files) == 0:
			sleep(0.1)
			continue 

		file = None
		file_ = np.random.choice(files)
		if file_.startswith('download'):
			file = file_
			

		if file == None:
			sleep(0.1)


		else:
			flag = True
			try:
				img = scipy.ndimage.imread(folder+"/"+file)
			except:
				flag = False


			if flag == True:
				print("Load success!")
				Popen("rm %s/*.png" % folder, shell=True).wait()

				shape = np.shape(img)
				target = np.zeros((shape[0],shape[1]), dtype=np.uint8)
				target = img[:,:,0] + img[:,:,1] + img[:,:,2]
				target[np.where(target>0)] = 255 

				Image.fromarray(target).save("%s/upload/target.png" % folder)
				Popen(copy_cmd % ("%s/upload/target.png" % folder), shell=True).wait()

			else:
				print("Load failed...")



