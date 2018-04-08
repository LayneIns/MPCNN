# coding: utf-8

import tensorflow as tf 
import model


def calculateAccuracy(score_list, label_list):
	total_case_number = 0
	right_case_number = 0
	one_case = []
	for i in range(len(score_list)):
		if label_list[i][1] == 1:
			if len(one_case) == 1:
				right_case_number += 1
			elif len(one_case) > 1:
				if one_case.index(max(one_case)) == 0:
					right_case_number += 1
			elif len(one_case) == 0:
				pass
			total_case_number += 1
			one_case = []

		one_case.append(score_list[i][1])

	if len(one_case) == 1:
		right_case_number += 1
	elif len(one_case) > 1:
		if one_case.index(max(one_case)) == 0:
			right_case_number += 1

	return total_case_number, right_case_number


def train(arg_config, training_data_mgr, valid_data_mgr, testing_data_mgr):
	train_model = model.MPCNN(arg_config)
	saver = tf.train.Saver(max_to_keep=1)

	with tf.Session() as sess:
		
		sess.run(train_model.init)

		for i in range(arg_config.epoch_num):
			training_data_mgr.initialize_batch_cnt()
			for j in range(0, training_data_mgr.total_batch, arg_config.batch_size):
				text1, text2, label = training_data_mgr.next_batch(arg_config.batch_size)

				_, loss, acc, scores = sess.run([train_model.train_op, train_model.loss, train_model.accuracy, train_model.scores], feed_dict={train_model.x1_input: text1, train_model.x2_input: text2, train_model.label: label})

				print "training --- epoch number:", str(i), ", batch:", str(j), ", loss:", str(loss), ", accuracy:", acc
				

				if j % (arg_config.batch_size * 60) == 0:
					total_score_list = []
					total_label = []
					valid_data_mgr.initialize_batch_cnt()
					for k in range(0, valid_data_mgr.total_batch, 128):
						text1, text2, label = valid_data_mgr.next_batch(128)
						total_label.extend(label.tolist())
						scores = sess.run(train_model.scores, feed_dict={train_model.x1_input: text1, train_model.x2_input: text2, train_model.label: label})
						total_score_list.extend(scores.tolist())

					total_case_number, right_case_number = calculateAccuracy(total_score_list, total_label)
					print "Validation: There are", total_case_number, "cases in the validation set,", right_case_number, "cases are right."
					print "Validation accuracy:", float(right_case_number)/total_case_number


					with open("./result/res.txt", "a") as fout:
						line1 = "Validation: There are " + str(total_case_number) + " cases in the validation set, " + str(right_case_number) + "cases are right.\n"
						line2 = "Validation accuracy:" + str(float(right_case_number)/total_case_number) + "\n"
						fout.write((line1+ line2 + "\n").encode('utf-8'))

				if j % (arg_config.batch_size * 60) == 0:
					total_score_list = []
					total_label = []
					testing_data_mgr.initialize_batch_cnt()
					for k in range(0, testing_data_mgr.total_batch, 128):
						text1, text2, label = testing_data_mgr.next_batch(128)
						total_label.extend(label.tolist())
						scores = sess.run(train_model.scores, feed_dict={train_model.x1_input: text1, train_model.x2_input: text2, train_model.label: label})
						total_score_list.extend(scores.tolist())

					total_case_number, right_case_number = calculateAccuracy(total_score_list, total_label)
					print "Testing: There are", total_case_number, "cases in the testing set,", right_case_number, "cases are right."
					print "Testing accuracy:", float(right_case_number)/total_case_number


					with open("./result/res.txt", "a") as fout:
						line1 = "Testing: There are " + str(total_case_number) + " cases in the testing set, " + str(right_case_number) + "cases are right.\n"
						line2 = "Testing accuracy:" + str(float(right_case_number)/total_case_number) + "\n"
						fout.write((line1+ line2 + "\n").encode('utf-8'))



