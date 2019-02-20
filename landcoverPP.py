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
import json



folder = sys.argv[1]
st = int(sys.argv[2])
ed = int(sys.argv[3])


file_list = os.listdir(folder)




def checkColor(img, target):
	sx = np.shape(img)[0]
	sy = np.shape(img)[1]

	t = np.zeros((sx,sy,3))
	m = np.zeros((sx,sy))

	t[:,:,0] = target[0]
	t[:,:,1] = target[1]
	t[:,:,2] = target[2]

	o = np.zeros((sx,sy),dtype=np.uint8)

	diff = np.abs(img - t)
	m = diff[:,:,0] + diff[:,:,1] + diff[:,:,2]

	o[np.where(m<20)] = 255 


	return o, np.sum(o)



# Urban land: 0,255,255 - Man-made, built up areas with human artifacts (can ignore roads for now which is hard to label)
# Agriculture land: 255,255,0 - Farms, any planned (i.e. regular) plantation, cropland, orchards, vineyards, nurseries, and ornamental horticultural areas; confined feeding operations.
# Rangeland: 255,0,255 - Any non-forest, non-farm, green land, grass
# Forest land: 0,255,0 - Any land with x% tree crown density plus clearcuts.
# Water: 0,0,255 - Rivers, oceans, lakes, wetland, ponds.
# Barren land: 255,255,255 - Mountain, land, rock, dessert, beach, no vegetation
# Unknown: 0,0,0 - Clouds and others

exist_list = [[],[],[],[],[],[]]
targets = [(0,255,255),(255,255,0),(255,0,255),(0,255,0),(0,0,255),(255,255,255)]
cc = 0

sorted(file_list)

for file in file_list:
	if file.endswith("mask.png"):
		cc = cc + 1

		if cc <= st or cc > ed :
			continue


		print(file)
		if cc % 10 == 0:
			print(cc,len(file_list))


		mask = scipy.ndimage.imread(folder+"/"+file)

		for i in xrange(6):
			o, s = checkColor(mask, targets[i])
			Image.fromarray(o).save(folder+"/"+file.replace('mask',"mask_%d"%i))
			if s > 255 * 16 * 16 :
				exist_list[i].append(file)



for i in xrange(6):
	print(i, len(exist_list[i]))

with open(folder+"/index.json", 'w') as outfile:
    json.dump(exist_list, outfile)


