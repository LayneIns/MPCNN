# coding: utf-8

import tensorflow as tf 
import model


def train(arg_config, training_data_mgr):
	train_model = model.MPCNN(arg_config)
	saver = tf.train.Saver(max_to_keep=1)

	with tf.Session() as sess:
		
		sess.run(model.init)

		for i in range(arg_config.epoch_num):
			training_data_mgr.initialize_batch_cnt()
			for j in range(0, training_data_mgr.total_batch, arg_config.batch_size):
				text1, text2, label = training_data_mgr.next_batch(arg_config.batch_size)

				_, loss, acc = sess.run([model.train_op, model.loss, model.accuracy], feed_dict={model.x1_input: text1, model.x2_input: text2, model.label: label})

				print "training --- epoch number:", str(i), ", batch:", str(j), ", loss:", str(loss), ", accuracy:", acc
