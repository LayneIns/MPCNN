# coding: utf-8

import utils
import numpy as np 
import data
import train


if __name__ == "__main__":
	
	print "Start to read training data..."
	training_filepath = "./data/training.txt"
	x1_text, x2_text, label_train = utils.loadDataAndLabels(training_filepath)
	print "\n"

	x_combined = x1_text + x2_text

	print "Get max length..."
	max_document_length = max([len(x1.split(" ")) for x1 in x_combined])
	print "max_document_length: ", max_document_length
	print "\n"

	print "Start to get word dictionary..."
	word_dict = utils.getWordDict(x_combined)
	print "There are", len(word_dict), "words in the training data."
	print "\n"

	print "Start to get one hot representation..."
	x_combined_vec = utils.convert(x_combined, word_dict, max_document_length)
	x1_train = np.asarray(x_combined_vec[:int(len(x_combined_vec)/2)])
	x2_train = np.asarray(x_combined_vec[int(len(x_combined_vec)/2):])
	print "\n"

	training_data_mgr = data.dataMgr(x1_train, x2_train, label_train)
	

	# ==========================================
	print "Start to read valid data..."
	valid_filepath = "./data/valid.txt"
	x1_text, x2_text, label_valid = utils.loadDataAndLabels(valid_filepath)
	print "There are", len(label_valid), "cases in the validation set."
	print "\n"

	x_combined = x1_text + x2_text

	print "Start to get one hot representation..."
	x_combined_vec = utils.convert(x_combined, word_dict, max_document_length)
	x1_train = np.asarray(x_combined_vec[:int(len(x_combined_vec)/2)])
	x2_train = np.asarray(x_combined_vec[int(len(x_combined_vec)/2):])
	print "\n"

	valid_data_mgr = data.dataMgr(x1_train, x2_train, label_valid)
	# ==========================================

	# ==========================================
	print "Start to read testing data..."
	testing_filepath = "./data/testing.txt"
	x1_text, x2_text, label_test = utils.loadDataAndLabels(testing_filepath)
	print "There are", len(label_test), "cases in the testing set."
	print "\n"

	x_combined = x1_text + x2_text

	print "Start to get one hot representation..."
	x_combined_vec = utils.convert(x_combined, word_dict, max_document_length)
	x1_train = np.asarray(x_combined_vec[:int(len(x_combined_vec)/2)])
	x2_train = np.asarray(x_combined_vec[int(len(x_combined_vec)/2):])
	print "\n"

	testing_data_mgr = data.dataMgr(x1_train, x2_train, label_test)
	# ==========================================





	arg_config = data.argConfig(max_document_length, len(word_dict))
	# print training_data_mgr.case_num[0:10]
	# print training_data_mgr.case_num[-10:]
	# print len(training_data_mgr.case_num)
	# raw_input()

	train.train(arg_config, training_data_mgr, valid_data_mgr, testing_data_mgr)
