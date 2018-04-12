# coding: utf-8

import tensorflow as tf 
import model
import random
from mpcnn2 import MPCNN
import sys


def calculateAccuracy(score_list, label_list):
	total_case_number = 0
	right_case_number = 0
	one_case = []
	for i in range(len(score_list)):
		if label_list[i][1] == 1:
			if len(one_case) == 1:
				right_case_number += 1
			elif len(one_case) > 1:
				if one_case.index(max(one_case)) == 0 and one_case.count(max(one_case)) == 1:
					right_case_number += 1
				elif one_case.index(max(one_case)) == 0 and one_case.count(max(one_case)) > 1:
					rand_int = random.randint(0, one_case.count(max(one_case))-1)
					if rand_int == 0:
						right_case_number += 1
			elif len(one_case) == 0:
				pass
			total_case_number += 1
			one_case = []

		one_case.append(score_list[i][1])

	if len(one_case) == 1:
		right_case_number += 1
	elif len(one_case) > 1:
		if one_case.index(max(one_case)) == 0 and one_case.count(max(one_case)) == 1:
			right_case_number += 1
		elif one_case.index(max(one_case)) == 0 and one_case.count(max(one_case)) > 1:
			rand_int = random.randint(0, one_case.count(max(one_case))-1)
			if rand_int == 0:
				right_case_number += 1

	return total_case_number, right_case_number


def train(arg_config, training_data_mgr, valid_data_mgr, testing_data_mgr):
	# train_model = model.MPCNN(arg_config)
	train_model = MPCNN(arg_config.max_length, arg_config.max_length, arg_config.num_classes, arg_config.vocab_size,
			arg_config.embedding_size, arg_config.filter_sizes, arg_config.num_filters, arg_config.num_filters)
	
	# saver = tf.train.Saver(max_to_keep=1)

	with tf.Session() as sess:
		
		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-3)
		grads_and_vars = optimizer.compute_gradients(train_model.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		sess.run(tf.global_variables_initializer())

		for i in range(arg_config.epoch_num):
			training_data_mgr.initialize_batch_cnt()
			for j in range(0, training_data_mgr.total_batch, arg_config.batch_size):
				text1, text2, label = training_data_mgr.next_batch(arg_config.batch_size)

				_, step, loss, acc= sess.run([train_op, global_step, train_model.loss, train_model.accuracy], feed_dict={train_model.input_x1: text1, train_model.input_x2: text2, train_model.input_y: label})

				print "training --- epoch number:", str(i), ", batch:", str(j), ", loss:", str(loss), ", accuracy:", acc
				
				current_step = tf.train.global_step(sess, global_step)

				if j % (arg_config.batch_size * 600) == 0:
					total_score_list = []
					total_label = []
					valid_data_mgr.initialize_batch_cnt()
					for k in range(0, valid_data_mgr.total_batch, 64):
						if k % 640 == 0:
							sys.stdout.flush()
							sys.stdout.write(" " * 30 + "\r")
							sys.stdout.flush()
							sys.stdout.write("processing: " + str(k) + "/" + str(valid_data_mgr.total_batch) + "\r")

						text1, text2, label = valid_data_mgr.next_batch(64)
						total_label.extend(label.tolist())
						scores = sess.run(train_model.scores, feed_dict={train_model.input_x1: text1, train_model.input_x2: text2, train_model.input_y: label})
						total_score_list.extend(scores.tolist())

					total_case_number, right_case_number = calculateAccuracy(total_score_list, total_label)
					print "Validation: There are", total_case_number, "cases in the validation set,", right_case_number, "cases are right."
					print "Validation accuracy:", float(right_case_number)/total_case_number


					with open("./result/res_2.txt", "a") as fout:
						line = "training --- epoch number: " + str(i) + ", batch: " + str(j) + "\n"
						line1 = "Validation: There are " + str(total_case_number) + " cases in the validation set, " + str(right_case_number) + " cases are right.\n"
						line2 = "Validation accuracy:" + str(float(right_case_number)/total_case_number) + "\n"
						fout.write((line + line1+ line2 + "\n").encode('utf-8'))

			total_score_list = []
			total_label = []
			testing_data_mgr.initialize_batch_cnt()
			for k in range(0, testing_data_mgr.total_batch, 64):
				sys.stdout.flush()
				sys.stdout.write(" " + "\r")
				sys.stdout.flush()
				sys.stdout.write(str(k) + "/" + str(testing_data_mgr.total_batch) + "\r")

				text1, text2, label = testing_data_mgr.next_batch(64)
				total_label.extend(label.tolist())
				scores = sess.run(train_model.scores, feed_dict={train_model.input_x1: text1, train_model.input_x2: text2, train_model.input_y: label})
				total_score_list.extend(scores.tolist())

			total_case_number, right_case_number = calculateAccuracy(total_score_list, total_label)
			print "Testing: There are", total_case_number, "cases in the testing set,", right_case_number, "cases are right."
			print "Testing accuracy:", float(right_case_number)/total_case_number


			with open("./result/res_2.txt", "a") as fout:
				line = "=========================================\nTesting: epoch: " + str(i) + "\n"
				line1 = "Testing: There are " + str(total_case_number) + " cases in the testing set, " + str(right_case_number) + "cases are right.\n"
				line2 = "Testing accuracy:" + str(float(right_case_number)/total_case_number) + "\n================================\n"
				fout.write((line1+ line2 + "\n").encode('utf-8'))



