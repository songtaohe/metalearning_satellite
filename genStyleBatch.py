import sys, getopt, os
from subprocess import Popen
import math
import numpy as np 
import scipy.misc
import scipy
import scipy.ndimage
import scipy.ndimage as nd
from PIL import Image

from time import time

import inputFilter



if __name__ == "__main__":
	tasks = []

	cityname = os.listdir(sys.argv[1])
	#output_folder = sys.argv[2]
	for name in cityname:

		if name == "boston" or name == "chicago" or name == "la":
			pass
		else:
			continue

		folder = sys.argv[1] + "/" + name + "/"


		# for tt in xrange(4):
		# 	tasks = []
		# 	for i in xrange(tt*9,tt*9+9):
		# 		fname = folder +"region%d_sat.png"%i  

		# 		tasks.append(Popen("python genStyle.py %s" % fname, shell=True))

		# 	for i in xrange(9):
		# 		tasks[i].wait()


		tasks = []
		for i in xrange(36):
			fname = folder +"region%d_sat.png"%i  

			tasks.append(Popen("python genStyle.py %s" % fname, shell=True))

		for i in xrange(36):
			tasks[i].wait()




	# 	with open(sys.argv[1] + "/" + name + "/boundingbox.txt", "r") as fin:
	# 		items = fin.readlines()[0].split(' ')

	# 		#print(name, items)

	# 		region = [float(x) for x in items]
			
	# 		print(name, region)

	# 		tasks.append([region, sys.argv[1] + "/" + name+"/", output_folder + "/"+ name +"/", sys.argv[1] + "/" + name + "/cityname.osm.pbf"])

	# for task in tasks:
	# 	Popen("python MetaTaskGenBatch.py %f %f %s %s %s "% (task[0][0], task[0][1], task[1], task[2], task[3]),shell=True).wait()



	


