# coding: utf-8

import numpy as np 
import re, sys

def clean_str(string):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`_]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()


def loadDataAndLabels(filepath):
	list_of_sentences1 = []
	list_of_sentences2 = []
	y = []
	k = 0
	with open(filepath) as training_file:
		for line in training_file:
			sys.stdout.write(" "*10 + "\r")
			sys.stdout.flush()
			sys.stdout.write(str(k) + "\r")
			sys.stdout.flush()
			k += 1
			parts = line.split('$$') #The two sentences and the similarity score are seperated by $$ delimiter
			list_of_sentences1.append(parts[0].strip())
			list_of_sentences2.append(parts[1].strip())
			if int(parts[2].strip()) > 0:
				prediction = [0, 1]
			else:
				prediction = [1, 0]
			y.append(prediction)

	x1_text = [clean_str(sent) for sent in list_of_sentences1]
	x2_text = [clean_str(sent) for sent in list_of_sentences2]
	# Generate labels
	y = np.asarray(y)

	return [x1_text,x2_text, y]


def getWordDict(sentence_list):
	word_dict = dict()
	word_dict['#UNK#'] = len(word_dict)
	word_dict['#head_entity#'] = len(word_dict)
	for sentence in sentence_list:
		words = [word.strip() for word in sentence.split() if word.strip()]
		for word in words:
			if word_dict.get(word, -1) == -1:
				word_dict[word] = len(word_dict)
	return word_dict


def padding(word_list, max_length):
	new_list = []
	for i in range(max_length):
		if i < len(word_list):
			new_list.append(word_list[i])
		else:
			new_list.append(0)
	return new_list


def convert(sentence_list, word_dict, max_length):
	new_sentence_list = []
	for sentence in sentence_list:
		new_sentence = [word.strip() for word in sentence.split() if word.strip()]
		new_sentence_vec = []
		for word in new_sentence:
			new_sentence_vec.append(word_dict.get(word, word_dict.get("#UNK#")))
		new_sentence_vec = padding(new_sentence_vec, max_length)

		new_sentence_list.append(new_sentence_vec)
	return new_sentence_list	
