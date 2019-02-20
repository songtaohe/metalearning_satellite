from PIL import Image 
from PIL import ImageFilter
import scipy.ndimage
import numpy as np
import scipy.fftpack as fp


## Functions to go from image to frequency-image and back
im2freq = lambda data: fp.rfft(fp.rfft(data, axis=0),
                               axis=1)
freq2im = lambda f: fp.irfft(fp.irfft(f, axis=1),
                             axis=0)

remmax = lambda x: x/x.max()
remmin = lambda x: x - np.amin(x, axis=(0,1), keepdims=True)
touint8 = lambda x: (remmax(remmin(x))*(256-1e-4)).astype(int)





def arr2im(data, fname):
    out = Image.new('RGB', data.shape[1::-1])
    out.putdata(map(tuple, data.reshape(-1, 3)))
    out.save(fname)


def applyFilter(img, t = 0):
	b = Image.fromarray(img)

	if t == 0:
		return img 

	if t == 1:
		return np.array(b.filter(ImageFilter.GaussianBlur(3)))

	if t == 2:
		return np.array(b.filter(ImageFilter.CONTOUR))

	if t == 3:
		return np.array(b.filter(ImageFilter.DETAIL))

	if t == 4:
		return np.array(b.filter(ImageFilter.EDGE_ENHANCE))

	if t == 5:
		return np.array(b.filter(ImageFilter.FIND_EDGES))

	if t == 6:
		return np.array(b.filter(ImageFilter.SMOOTH_MORE))

	if t == 7:

		fMask1 = np.array(Image.open('filters/highpass.png')).astype(float) / 255
		fMask = np.zeros((4096,4096,3))
		fMask[:,:,0] = fMask1
		fMask[:,:,1] = fMask1
		fMask[:,:,2] = fMask1

		return touint8(freq2im(im2freq(img) * fMask)).astype(np.uint8)

	if t == 8:

		fMask1 = np.array(Image.open('filters/bandpass3.png')).astype(float) / 255
		fMask = np.zeros((4096,4096,3))
		fMask[:,:,0] = fMask1
		fMask[:,:,1] = fMask1
		fMask[:,:,2] = fMask1

		return touint8(freq2im(im2freq(img) * fMask)).astype(np.uint8)

	if t == 9:

		fMask1 = np.array(Image.open('filters/bandpass4.png')).astype(float) / 255
		fMask = np.zeros((4096,4096,3))
		fMask[:,:,0] = fMask1
		fMask[:,:,1] = fMask1
		fMask[:,:,2] = fMask1

		return touint8(freq2im(im2freq(img) * fMask)).astype(np.uint8)






if __name__ == "__main__":
	a  = scipy.ndimage.imread('region12_sat.png')

	b = Image.fromarray(a)

	fMask1 = np.array(Image.open('filters/highpass.png')).astype(float) / 255
	fMask = np.zeros((4096,4096,3))
	fMask[:,:,0] = fMask1
	fMask[:,:,1] = fMask1
	fMask[:,:,2] = fMask1


	arr2im(touint8(freq2im(im2freq(np.array(b)) * fMask)), "testFFT.png")

	exit()


	b.filter(ImageFilter.GaussianBlur(2)).save('test1.png')
	b.filter(ImageFilter.GaussianBlur(3)).save('test1.png')
	b.filter(ImageFilter.CONTOUR).save('test2.png')
	b.filter(ImageFilter.DETAIL).save('test3.png')
	b.filter(ImageFilter.EDGE_ENHANCE).save('test4.png')
	#b.filter(ImageFilter.EDGE_ENHANCE_MORE).save('test5.png')
	#b.filter(ImageFilter.EMBOSS).save('test6.png')
	b.filter(ImageFilter.FIND_EDGES).save('test7.png')
	#b.filter(ImageFilter.SMOOTH).save('test8.png')
	b.filter(ImageFilter.SMOOTH_MORE).save('test9.png')
	b.filter(ImageFilter.SHARPEN).save('test10.png')



