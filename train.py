# coding: utf-8

import tensorflow as tf 
import model


def train(arg_config, training_data_mgr):
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
				print scores

				#if j % (arg_config.batch_size * 15) == 0:

