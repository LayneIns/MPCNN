# coding: utf-8

import numpy as np 


class argConfig:
	def __init__(self, max_document_length, vocab_size):
		self.epoch_num = 10
		self.max_length = max_document_length
		self.vocab_size = vocab_size
		self.num_classes = 2
		self.embedding_size = 300
		self.filter_sizes = [1, 2, 3]
		self.num_filters = 128
		self.learning_rate = 0.001
		self.batch_size = 32

class dataMgr:

	def __init__(self, text1, texe2, label):
		self.text1 = text1
		self.text2 = text2
		self.label = label
		self.total_batch = len(text1)
		self.batch_cnt = 0

	def initialize_batch_cnt(self):
		self.batch_cnt = 0

	def next_batch(self, batch_size):
		if self.batch_cnt < self.total_batch - batch_size:
			text1 = self.text1[self.batch_cnt: self.batch_cnt+batch_size]
			text2 = self.text2[self.batch_cnt: self.batch_cnt+batch_size]
			label = self.label[self.batch_cnt: self.batch_cnt+batch_size]

			self.batch_cnt += batch_size
		else self.batch_cnt < self.total_batch:
			text1 = self.text1[self.batch_cnt: ]
			text2 = self.text2[self.batch_cnt: ]
			label = self.label[self.batch_cnt: ]

			self.batch_cnt = self.total_batch
		

		return text1, text2, label
