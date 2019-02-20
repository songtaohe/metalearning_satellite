

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
folder_3 = sys.argv[3]


folder_output = sys.argv[4]


for rid in xrange(36):
	channel_0 = nd.imread(folder_1+"region%d_output.png"%rid)
	channel_1 = nd.imread(folder_2+"region%d_output.png"%rid)
	channel_2 = nd.imread(folder_3+"region%d_output.png"%rid)

	dims = np.shape(channel_0)

	output = np.zeros((dims[0],dims[1],3),dtype=np.uint8)

	output[:,:,0]=channel_0
	output[:,:,1]=channel_1
	output[:,:,2]=channel_2


	Image.fromarray(output).save(folder_output+"region%d_mix.png"%rid)

	output = scipy.misc.imresize(output,(dims[0]/2, dims[1]/2), mode="RGB")
	Image.fromarray(output).save(folder_output+"region%d_mix2x.png"%rid)

	output = scipy.misc.imresize(output,(dims[0]/4, dims[1]/4), mode="RGB")
	Image.fromarray(output).save(folder_output+"region%d_mix4x.png"%rid)

	output = scipy.misc.imresize(output,(dims[0]/8, dims[1]/8), mode="RGB")
	Image.fromarray(output).save(folder_output+"region%d_mix8x.png"%rid)





