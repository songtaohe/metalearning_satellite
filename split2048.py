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




def split2048(name1, name2, rgb=False):
	base = scipy.ndimage.imread(name1+name2)

	if rgb == True:
		Image.fromarray(base[0:2048,0:2048,:]).save(name1+"_0"+name2)
		Image.fromarray(base[2048:4096,0:2048,:]).save(name1+"_1"+name2)
		Image.fromarray(base[0:2048,2048:4096,:]).save(name1+"_2"+name2)
		Image.fromarray(base[2048:4096,2048:4096,:]).save(name1+"_3"+name2)
	else:
		Image.fromarray(base[0:2048,0:2048]).save(name1+"_0"+name2)
		Image.fromarray(base[2048:4096,0:2048]).save(name1+"_1"+name2)
		Image.fromarray(base[0:2048,2048:4096]).save(name1+"_2"+name2)
		Image.fromarray(base[2048:4096,2048:4096]).save(name1+"_3"+name2)


folder = sys.argv[1]
for i in xrange(36):
	print(i)
	split2048(folder+"/region%d"%i, "_sat.png", rgb=True)
	split2048(folder+"/region%d"%i, "_t1.png", rgb=False)
	split2048(folder+"/region%d"%i, "_t2.png", rgb=False)
	split2048(folder+"/region%d"%i, "_t3.png", rgb=False)
	split2048(folder+"/region%d"%i, "_t4.png", rgb=False)
	split2048(folder+"/region%d"%i, "_t5.png", rgb=False)