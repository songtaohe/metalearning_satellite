

import sys
from PIL import Image
import scipy
import random
import numpy as np 
from time import time, sleep
from subprocess import Popen
import scipy.ndimage as nd
import json
import scipy



folder_1 = sys.argv[1]
folder_2 = sys.argv[2]

folder_output = sys.argv[3]

def ExtractBoundary(img, threshold=0.5, scale=0.125, max_v = 1.0):
	neighbors = nd.convolve((img>threshold).astype(np.int),[[1,1,1],[1,0,1],[1,1,1]],mode='constant',cval=0.0)
	img = np.copy(img)
	img = img * scale
	img[np.where((neighbors>0)&(neighbors<8))] = max_v		
	return img 



for rid in xrange(36):
	print(rid)
	sat_img = nd.imread(folder_1+"region%d_sat.png"%rid)
	channel_1 = nd.imread(folder_2+"region%d_output.png"%rid)
	
	dim = np.shape(channel_1)

	sat_img = scipy.misc.imresize(sat_img,(dim[0],dim[1]),mode="RGB")




	sat_img = sat_img.astype(np.int)  

	sat_img[:,:,0] += ExtractBoundary(channel_1,threshold=128,max_v=255).astype(int)
	sat_img[:,:,1] += ExtractBoundary(channel_1,threshold=128,max_v=255).astype(int)

	sat_img = np.clip(sat_img,0,255)


	Image.fromarray(sat_img.astype(np.uint8)).save(folder_output+"region%d_output.png"%rid)

	



