from MAML import *
import sys
import tensorflow as tf

if __name__ == "__main__":
	dataloader = DataLoader('/data/songtao/metalearning/dataset/boston_task5_small/', 16, 5, preload = False)


	testCases = dataloader.getTestBatchs(5,20)


	with tf.Session() as sess:
		model = MAML(sess)
		model.restoreModel(sys.argv[1])

		
		for i in xrange(5):
			eeid = 0

			testInputs = [testCases[0][i],testCases[1][i], testCases[2][i],testCases[3][i],i]

			result, result_debug = model.runModel(testInputs[0],testInputs[1],testInputs[2])
			tmp_img = np.zeros((512,512,3), dtype=np.uint8)
			tmp_img2 = np.zeros((512,512), dtype=np.uint8)

			for ind in xrange(len(testInputs[2])):
				tmp_img = (testInputs[2][ind,:,:,:]*255).reshape(512,512,3).astype(np.uint8)
				Image.fromarray(tmp_img).save('e2/img_%d_%d_sat.png' % (eeid, ind))
				if ind < 5:
					print(np.amin(result[ind,:,:,:]*255), np.amax(result[ind,:,:,:]*255))

				tmp_img2 = (result[ind,:,:,0]*255).reshape(512,512).astype(np.uint8)
				Image.fromarray(tmp_img2).save('e2/img_%d_%d_task%d_output.png' % (eeid, ind, testInputs[4]))
				if ind == 0:
					print(tmp_img2)

				tmp_img2 = (result[ind,:,:,0]*255).reshape(512,512)
				#tmp_img2 = (result_debug[ind,:,:,0]*255).reshape(512,512)
				tmp_img2[np.where(tmp_img2>127)] = 255
				tmp_img2[np.where(tmp_img2<=127)] = 0

				# if ind == 0:
				# 	print(tmp_img2)


				if ind < 5:
					print(np.amax(tmp_img2),np.amin(tmp_img2))

				tmp_img2 = tmp_img2.astype(np.uint8)

				Image.fromarray(tmp_img2).save('e2/img_%d_%d_task%d_output_sharp.png' % (eeid, ind, testInputs[4]))

				tmp_img2 = (testInputs[3][ind,:,:,0]*255).reshape(512,512).astype(np.uint8)
				Image.fromarray(tmp_img2).save('e2/img_%d_%d_task%d_target.png' % (eeid, ind, testInputs[4]))
				
