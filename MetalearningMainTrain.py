import argparse

from MetalearningModels import MAMLFirstOrder
import MetalearningLoader 
from time import time,sleep
from subprocess import Popen
import random
import scipy
from PIL import Image
import os, datetime
import numpy as np
import tensorflow as tf


parser = argparse.ArgumentParser()

parser.add_argument('-s', action='store', dest='model_save_name', type=str,
                    help='model save folder', required=True)

parser.add_argument('-run', action='store', dest='run_name', type=str,
                    help='run name', required=True)


parser.add_argument('-model', action='store', dest='model_type', type=str,
                    help='model type', required=True)


parser.add_argument('-r', action='store', dest='model_recover', type=str,
                    help='path to model for recovering')

parser.add_argument('-d', action='store', dest='data_folder', type=str,
                    help='dataset folder', default='/data/songtao/')


parser.add_argument('-lr', action='store', dest='learning_rate', type=float,
                    help='dataset folder', default=0.00005)

# inner_lr = 0.001 
# inner_lr_testing = 0.001 

args = parser.parse_args()

print(args)


if __name__ == "__main__":
	name = args.model_save_name
	run = name + args.run_name

	Popen("mkdir -p %s" % name, shell=True).wait()

	output_folder1 = name+"/e1/"
	output_folder2 = name+"/e2/"
	model_folder = name+"/model/"
	#log_folder = name+"/log"
	log_folder = "alllogs/log"

	Popen("mkdir -p %s" % output_folder1, shell=True).wait()
	Popen("mkdir -p %s" % output_folder2, shell=True).wait()
	Popen("mkdir -p %s" % model_folder, shell=True).wait()
	Popen("mkdir -p %s" % log_folder, shell=True).wait()


	random.seed(321)

	with tf.Session() as sess:

		model = MAMLFirstOrder(sess)
		model.meta_lr_val = args.learning_rate

		if args.model_recover is not None:
			model.restoreModel(args.model_recover)

		writer = tf.summary.FileWriter(log_folder+"/"+run, sess.graph)
		Popen("pkill tensorboard", shell=True).wait()
		sleep(1)
		print("tensorboard --logdir=%s"%log_folder)
		logserver = Popen("tensorboard --logdir=%s"%log_folder, shell=True)


		print("dataset folder", args.data_folder)

		dataloader = MetalearningLoader.GetDataLoader(args.data_folder)
		task_p = [1.0/len(dataloader.loaders) for x in xrange(len(dataloader.loaders))]

		loaders = dataloader.loaders

		print("Preload")
		t0 = time()
		dataloader.preload(100)
		print("Preload Done", time()-t0)


		t0 = time()
		testCase = [dataloader.loadBatchFromTask(i,5,10) for i in xrange(len(loaders))]
		testCase += [dataloader.loadBatchFromTask(i,5,10) for i in xrange(len(loaders))]
		print("Sample Test Data Done", time()-t0)


		step = 0

		losses = {}
		loss_curves = {}

		ts = time()

		last_longterm_loss = 10000
		longterm_loss = 0
		test_loss = 0
		sum_g1 = 0
		sum_g2 = 0

		grad_scale = 1.0


		while True:
			
			if step % 500 == 0 and step != 0:
				t0 = time()
				dataloader.preload(200)
				print("Step ", step, "Preload Done", time()-t0)


			iA,oA, iB, oB, task_id = dataloader.loadBatch(5,1,p=task_p)

			ret = model.trainModel(iA,oA,iB,oB, 1.0)

			loss = ret[1]
			loss_curve = ret[2:]

			if task_id in loss_curves:
				loss_curves[task_id][0] = [loss_curves[task_id][0][ind] + loss_curve[ind] for ind in xrange(len(loss_curve))]
				loss_curves[task_id][1] += 1
			else:
				loss_curves[task_id] = [loss_curve,1]

			
			#print(step,loss)
			step = step + 1

			if task_id in losses:
				losses[task_id][0] += loss
				losses[task_id][1] += 1

			else:
				losses[task_id] = [loss,1]

			if step % 10 == 0:
				print(step)
			if step % 200 == 0 or step == 1:
				train_loss = 0
				if step != 1:
					ss = 0
					cc = 0
					s = 0
					for t in xrange(len(loaders)):
						if t in losses:
							print(t, losses[t][0]/losses[t][1])
							cc += losses[t][1]
							s += losses[t][0]

						if t in loss_curves:
							ppp = [loss_curves[t][0][ind]/loss_curves[t][1] for ind in xrange(len(loss_curves[t][0]))]
							print('curve', t, ppp)

					losses = {}
					loss_curves = {}

					print(step, "total loss", s/cc)
					train_loss = s/cc
					longterm_loss += s/cc

					if step % 400 == 0:
						model.saveModel(model_folder+"/model%d"%step)

				eeid = 0
	
				#test_num = 2

				# if step % 1000 == 0:
				# 	tmpTestCase = testCase
				# else:
				# 	tmpTestCase = testCase[:test_num]


				if step % 400 == 0 or step < 400:
					test_loss = 0
					#test_loss_first_two = 0

					eeid = 0
					for testCases in testCase:

						result, result_loss, _ = model.runModel(testCases[0],testCases[1],testCases[2], testCases[3])

						tmp_img = np.zeros((512,512,3), dtype=np.uint8)
						tmp_img2 = np.zeros((512,512), dtype=np.uint8)

						# output input
						for ind in xrange(len(testCases[0])):
							tmp_img = (testCases[0][ind,:,:,:]*255).astype(np.uint8)
							Image.fromarray(tmp_img).save(output_folder1+'img_%d_%d_sat.png' % (eeid, ind))

							tmp_img2 = (testCases[1][ind,:,:,0]*255).astype(np.uint8)
							Image.fromarray(tmp_img2).save(output_folder1+'img_%d_%d_target.png' % (eeid, ind))

						for ind in xrange(len(testCases[2])):
							tmp_img = (testCases[2][ind,:,:,:]*255).astype(np.uint8)
							Image.fromarray(tmp_img).save(output_folder1+'img_%d_%d_sat.png' % (eeid, ind+len(testCases[0])))

							tmp_img2 = (testCases[3][ind,:,:,0]*255).astype(np.uint8)
							Image.fromarray(tmp_img2).save(output_folder1+'img_%d_%d_target.png' % (eeid, ind+len(testCases[0])))

							tmp_img2 = (result[ind,:,:,0]*255).astype(np.uint8)
							Image.fromarray(tmp_img2).save(output_folder1+'img_%d_%d_output.png' % (eeid, ind+len(testCases[0])))

						test_loss = test_loss + result_loss
						eeid += 1

					test_loss/=eeid
				

				print("test_loss", test_loss)

				if step>1:
					summary = model.addLog(test_loss, train_loss, model.meta_lr_val)
					writer.add_summary(summary, step)
				

				if step % 5000 == 0:



					print("longterm_loss",longterm_loss/5," lr is", model.meta_lr_val)

					longterm_loss = test_loss

					print(longterm_loss, last_longterm_loss)

					if longterm_loss > last_longterm_loss*1.0:
						#reduce learning rate
						#if model.meta_lr_val > 0.0001:
						if model.meta_lr_val > 0.000005:
							#model.meta_lr_val = model.meta_lr_val / 2
							longterm_loss = 10000 # don't reduce the learning rate two times
						# new_lr = sess.run([model.change_lr_op], feed_dict = {model.meta_lr_val:model.meta_lr_val})

						print("change lr to", model.meta_lr_val)

					last_longterm_loss = longterm_loss
					longterm_loss = 0

					if step % 5000 == 0:
						model.meta_lr_val = model.meta_lr_val / 1.1
						print("change lr to", model.meta_lr_val)


				t_int = time() - ts 
				ts = time()
				print("iterations per hour", 3600/t_int * 200)
				#print("iterations per hour", 3600/t_int * 200, t_int, dataloader.total_time, dataloader.total_time/t_int)
				#dataloader.total_time = 0

	
		writer.close()



























