from PIL import Image
import scipy.ndimage
import sys 
import numpy as np 

a = scipy.ndimage.imread(sys.argv[1])
c = int(sys.argv[2])

o = np.zeros((np.shape(a)[0:2]))
t = a[:,:,c]

o = t 

o[np.where(o>0)] = 255

Image.fromarray(o).save(sys.argv[3]) 
