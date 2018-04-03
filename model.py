# coding: utf-8

import tensorflow as tf 

class MPCNN:
	def __init__(self, arg_config):
		self.x1_input = tf.placeholder(tf.int32, [None, arg_config.max_length], name="input_x1") # [batch_size, max_length]
		self.x2_input = tf.placeholder(tf.int32, [None, arg_config.max_length], name="input_x2") # [batch_size, max_length]
		self.label = tf.placeholder(tf.int32, [None, arg_config.num_classes], name="label") # [batch_size, num_classses]

		with tf.variable_scope("embedding_layer"):
			W = tf.get_variable("W", [arg_config.vocab_size, arg_config.embedding_size], initializer=tf.random_normal_initializer(mean=0, stddev=1))
			self.x1_embedding = tf.nn.embedding_lookup(W, self.x1_input, name="x1_embedding") # [batch_size, max_length, embedding_size]
			self.x2_embedding = tf.nn.embedding_lookup(W, self.x2_input, name="x2_embedding") # [batch_size, max_length, embedding_size]

		self.x1_embedding_expand = tf.expand_dims(self.x1_embedding, axis=-1, name="x1_embedding_expand") # [batch_size, max_length, embedding_size, 1]
		self.x2_embedding_expand = tf.expand_dims(self.x2_embedding, axis=-1, name="x2_embedding_expand") # [batch_size, max_length, embedding_size, 1]

		# block A
		x1_avg_pooled_output_groupA = []
		x1_min_pooled_output_groupA = []
		x1_max_pooled_output_groupA = []
		x2_avg_pooled_output_groupA = []
		x2_min_pooled_output_groupA = []
		x2_max_pooled_output_groupA = []

		# block B
		x1_min_pooled_output_groupB = []
		x1_max_pooled_output_groupB = []
		x2_min_pooled_output_groupB = []
		x2_max_pooled_output_groupB = []


		all_type_x2_pools_groupA =[]
		all_type_x1_pools_groupA =[]

		all_type_x2_pools_groupB =[]
		all_type_x1_pools_groupB =[]

		for i, filter_size in enumerate(arg_config.filter_sizes): # filter_size-> [1, 2, 3]
			with tf.variable_scope("block_A-%s" % filter_size):
				with tf.name_scope("embedded_x1_expanded_blockA"):
					x1_avgpooled, x1_maxpooled, x1_minpooled = \
							self.group_A(filter_size, arg_config.embedding_size, \
										arg_config.num_filters, self.x1_embedding_expand, \
										arg_config.max_length) # [batch_size, num_filters]

				with tf.name_scope("embedded_x2_expanded_blockA"):
					x2_avgpooled, x2_maxpooled, x2_minpooled = \
							self.group_A(filter_size, arg_config.embedding_size, \
										arg_config.num_filters, self.x2_embedding_expand, \
										arg_config.max_length, reuse_flag = True) # [batch_size, num_filters]
			
				x1_avg_pooled_output_groupA.append(x1_avgpooled)
				x1_max_pooled_output_groupA.append(x1_maxpooled)
				x1_min_pooled_output_groupA.append(x1_minpooled)
				x2_avg_pooled_output_groupA.append(x2_avgpooled)
				x2_max_pooled_output_groupA.append(x2_maxpooled)
				x2_min_pooled_output_groupA.append(x2_minpooled)
				# each shape is [3, batch_size, num_filters]

			with tf.variable_scope("block_B-%s" % filter_size):
				x1_maxpool_perdimension = []
				x1_minpool_perdimension = []
				x2_maxpool_perdimension = []
				x2_minpool_perdimension = []
				for embedding_idx in range(arg_config.embedding_size):
					embedded_x1_expanded_shape = tf.shape(self.x1_embedding_expand)
					shape0 = embedded_x1_expanded_shape[0]
					shape1 = embedded_x1_expanded_shape[1]
					shape3 = embedded_x1_expanded_shape[3]
					embedding_slice_x1 = tf.slice(self.x1_embedding_expand,[0, 0, embedding_idx, 0], [shape0, shape1, 1, shape3], name="embedding_slice")
					# shape: [batch_size, max_length, 1, 1]
					embedding_slice_x2 = tf.slice(self.x2_embedding_expand, [0, 0, embedding_idx, 0],[shape0, shape1, 1, shape3], name="embedding_slice")
					# shape: [batch_size, max_length, 1, 1]
					with tf.variable_scope("embedding_scope_%s" % embedding_idx):
						_, x1_maxpooled, x1_minpooled = self.group_A(filter_size, 1, arg_config.num_filters, embedding_slice_x1, arg_config.max_length)
						_, x2_maxpooled, x2_minpooled = self.group_A(filter_size, 1, arg_config.num_filters, embedding_slice_x2, arg_config.max_length, reuse_flag=True)

					x1_maxpool_perdimension.append(x1_maxpooled)  # [embed_dim x (MB x Num_filter)]
					x1_minpool_perdimension.append(x1_minpooled)

					x2_maxpool_perdimension.append(x2_maxpooled)
					x2_minpool_perdimension.append(x2_minpooled)
					# shape:[embedding_size, batch_size, num_filters]

				x1_max_pooled_output_groupB.append(tf.transpose(tf.stack(x1_maxpool_perdimension), perm=[1,0,2], name ="transpose_x1_maxpool_perdimension_102") )   #[types_of_filter_sizes x (MB x embed_dim x Num_filter)]
				# after tf.stack: [embedding_size, batch_size, num_filters]
				# after tf.transpose: [batch_size, embedding_size, num_filters]
				# after append: [3, batch_size, embedding_size, num_filters]
				x1_min_pooled_output_groupB.append(tf.transpose(tf.stack(x1_minpool_perdimension), perm=[1,0,2]) )

				x2_max_pooled_output_groupB.append(tf.transpose(tf.stack(x2_maxpool_perdimension), perm=[1,0,2]) )
				x2_min_pooled_output_groupB.append(tf.transpose(tf.stack(x2_minpool_perdimension), perm=[1,0,2]) )

		all_type_x1_pools_groupA.append(x1_avg_pooled_output_groupA)
		all_type_x1_pools_groupA.append(x1_max_pooled_output_groupA)
		all_type_x1_pools_groupA.append(x1_min_pooled_output_groupA)

		all_type_x2_pools_groupA.append(x2_avg_pooled_output_groupA)
		all_type_x2_pools_groupA.append(x2_max_pooled_output_groupA)
		all_type_x2_pools_groupA.append(x2_min_pooled_output_groupA)
		# shape is [ types_of_poolings(3), types_of_filter_sizes(3), batch_size, num_filters]



		all_type_x1_pools_groupB.append(x1_max_pooled_output_groupB)
		all_type_x1_pools_groupB.append(x1_min_pooled_output_groupB)

		all_type_x2_pools_groupB.append(x2_max_pooled_output_groupB)
		all_type_x2_pools_groupB.append(x2_min_pooled_output_groupB)
		# [types_of_poolings(2), types_of_filter_sizes(3), batch_size, embedding_size, num_filters]

		feah = self.algorithm_1(all_type_x1_pools_groupA, all_type_x2_pools_groupA)
		# shape [batch_size, 3*num_filters*2]

		feaa,feab = self.algorithm_2(all_type_x1_pools_groupA, all_type_x2_pools_groupA, all_type_x1_pools_groupB, all_type_x2_pools_groupB)
		# shape [batch_size, 3*ws1*ws2*3] 
		# shape [batch_size, 2*ws1*3*num_filters]

		number_of_filter_windows = len(arg_config.filter_sizes)
		fea_shape_1 =int( (3*2*arg_config.num_filters) + \
							(3*number_of_filter_windows*number_of_filter_windows*3) + \
							(2*number_of_filter_windows*arg_config.num_filters*3)
						)

		fea = tf.concat([feah,feaa,feab],-1)
		# shape [batch_size, fea_shape_1]

		weights = {
			'hidden1': tf.get_variable("hidden1_w",[fea_shape_1, arg_config.hidden_num_units]),
			'hidden2': tf.get_variable("hidden2_w",[arg_config.hidden_num_units, arg_config.hidden_num_units]),
			'output': tf.get_variable("output_w",[arg_config.hidden_num_units, arg_config.num_classes]),
		}

		biases = {
			'hidden1': tf.get_variable("hidden1_b",[arg_config.hidden_num_units]),
			'hidden2': tf.get_variable("hidden2_b",[arg_config.hidden_num_units]),
			'output': tf.get_variable("output_b",[arg_config.num_classes])
		}

		hidden_layer1 = tf.add(tf.matmul(fea, weights['hidden1']), biases['hidden1'])
		activated_hidden_layer1 = tf.tanh(hidden_layer1)
		hidden_layer2 = tf.add(tf.matmul(activated_hidden_layer1, weights['hidden2']), biases['hidden2'])
		activated_hidden_layer2 = tf.tanh(hidden_layer2)

		with tf.name_scope("output"):
			self.scores = tf.nn.xw_plus_b(activated_hidden_layer2, weights['output'], biases['output'], name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")

		# CalculateMean cross-entropy loss
		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.label)
			self.loss = tf.reduce_mean(losses) 

		# Accuracy
		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

		self.optimizer = tf.train.AdamOptimizer(learning_rate=arg_config.learning_rate)
		self.train_op = self.optimizer.minimize(self.loss)		
		
		self.init = tf.global_variables_initializer()



	def group_A(self, filter_size, embedding_size, num_filters, embedding_expand, max_length, reuse_flag=False):
		with tf.variable_scope("group_A_filters_biases", reuse=reuse_flag):
			filter_shape = [filter_size, embedding_size, 1, num_filters]
			filter_avgpool = tf.get_variable("filter_avgpool", filter_shape)
			filter_minpool = tf.get_variable("filter_minpool", filter_shape)
			filter_maxpool = tf.get_variable("filter_maxpool", filter_shape)

			b_minpool = tf.get_variable("b_minpool", initializer=0.1*tf.ones([num_filters]))
			b_maxpool = tf.get_variable("b_maxpool", initializer=0.1*tf.ones([num_filters]))
			b_avgpool = tf.get_variable("b_avgpool", initializer=0.1*tf.ones([num_filters]))

		conv_avgpool = tf.nn.conv2d(embedding_expand, \
				filter_avgpool, \
				strides=[1, 1, 1, 1], \
				padding='VALID', \
				name="conv_avgpool") # shape: [batch_size, max_length-filter_size+1, 1, num_filters]
		h_avgpool = tf.nn.tanh(tf.nn.bias_add(conv_avgpool, b_avgpool), name="tanh_avgpool")

		avgpooled = tf.nn.avg_pool(h_avgpool, \
			ksize=[1, max_length-filter_size+1, 1, 1], \
			strides=[1, 1, 1, 1], \
			padding='VALID', \
			name="avgpool") # shape: [batch_size, 1, 1, num_filters]

		conv_maxpool = tf.nn.conv2d(embedding_expand, \
				filter_maxpool, \
				strides=[1, 1, 1, 1], \
				padding='VALID', \
				name="conv_maxpool") # shape: [batch_size, max_length-filter_size+1, 1, num_filters]
		h_maxpool = tf.nn.tanh(tf.nn.bias_add(conv_maxpool, b_maxpool), name="tanh_maxpool")

		maxpooled = tf.nn.max_pool(h_maxpool, \
			ksize=[1, max_length-filter_size+1, 1, 1], \
			strides=[1, 1, 1, 1], \
			padding='VALID', \
			name="maxpool") # shape: [batch_size, 1, 1, num_filters]

		conv_minpool = tf.nn.conv2d(embedding_expand, \
				filter_minpool, \
				strides=[1, 1, 1, 1], \
				padding='VALID', \
				name="conv_minpool") # shape: [batch_size, max_length-filter_size+1, 1, num_filters]
		h_minpool = tf.nn.tanh(tf.nn.bias_add(conv_minpool, b_minpool), name="tanh_minpool")

		minpooled = -tf.nn.max_pool(-h_minpool, \
			ksize=[1, max_length-filter_size+1, 1, 1], \
			strides=[1, 1, 1, 1], \
			padding='VALID', \
			name="minpool") # shape: [batch_size, 1, 1, num_filters]

		avgpooled_squeezed = tf.squeeze(avgpooled, axis = [1,2]) # shape is (batch_size x num_Filters)
		maxpooled_squeezed = tf.squeeze(maxpooled, axis = [1,2])
		minpooled_squeezed = tf.squeeze(minpooled, axis = [1,2])

		return avgpooled_squeezed, maxpooled_squeezed, minpooled_squeezed


	def algorithm_1(self, all_type_x1_pools_groupA, all_type_x2_pools_groupA):
		with tf.name_scope("algorithm_1"):
			feah = []
			for p in range(len(all_type_x1_pools_groupA)):
				x1_sentences_stacked = tf.stack(all_type_x1_pools_groupA[p], name="stack_x1_sentences")
				# shape [types_of_filters_sizes, batch_size, num_filters]
				x1_sentences_stacked = tf.transpose(x1_sentences_stacked, perm=[1,2,0])
				# shape [batch_size, num_filters, types_of_filters_sizes]
				x2_sentences_stacked = tf.stack(all_type_x2_pools_groupA[p], name="stack_x2_sentences")
				x2_sentences_stacked = tf.transpose(x2_sentences_stacked, perm=[1,2,0])
				x1_sentence_normalised = self.normalise(x1_sentences_stacked)
				x2_sentence_normalised = self.normalise(x2_sentences_stacked)
				# shape: [batch_size, num_filters, types_of_filters_sizes]


				cosine_dis = tf.squeeze(cosine_distance(x1_sentence_normalised, x2_sentence_normalised, dim=-1), axis=-1)
				# shape [batch_size, num_filters]
				l2diff = tf.sqrt(tf.reduce_sum(tf.square(tf.clip_by_value(tf.subtract(x1_sentences_stacked, x2_sentences_stacked, name="subtract_inp1_inp2"), 1e-7, 1e+10)), axis=-1))
				feah.append(tf.concat([cosine_dis, l2diff], axis=-1))
				# after loop: shape [3, batch_size, num_filters * 2]


			feah_tensor = tf.transpose(tf.stack(feah), [1, 0, 2])
			# shape [batch_size, 3, num_filters * 2]
			feah_tensor = tf.reshape(feah_tensor, [tf.shape(feah_tensor)[0], -1])
			# shape [batch_size, 3*num_filters*2]
		return feah_tensor

	def algorithm_2(self, Ga_inp1_sentences, Ga_inp2_sentences, Gb_inp1_sentences, Gb_inp2_sentences):
		# shape is [ 3, 3, batch_size, num_filters]
		# [ 2, 3, batch_size, embedding_size, num_filters]
		with tf.name_scope("Algo2"):
			feaa = []
			feab = []
			for p in range(len(Ga_inp1_sentences)):  # for each pooling type p:
				for ws1 in range(len(Ga_inp1_sentences[p])):
					oG1a = Ga_inp1_sentences[p][ws1] #shape is MB xNum_Filters_for_each_size
					for ws2 in range(len(Ga_inp2_sentences[p])):
						oG2a = Ga_inp2_sentences[p][ws2] #shape is MB xNum_Filters_for_each_size
						oG1a_normalised = self.normalise(oG1a)
						oG2a_normalised = self.normalise(oG2a)

						displacement  = tf.sqrt(tf.reduce_sum(tf.square(tf.clip_by_value(tf.subtract(oG1a, oG2a),1e-7,1e+10)), -1, keep_dims=True))
						cd = cosine_distance(oG1a_normalised, oG2a_normalised, dim=-1)
						l2diff = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(tf.clip_by_value(tf.subtract(oG1a, oG2a), 1e-7, 1e+10)), axis=-1)),-1)
						feaa.append(tf.concat([cd, l2diff,displacement], axis=-1))

			for p2 in range(len(Gb_inp1_sentences)):  # for each pooling type p2:
				for ws1 in range(len(Gb_inp1_sentences[p2])):
					oG1b = Gb_inp1_sentences[p2][ws1] #shape is MB x embed_dim x Num_filter
					oG2b = Gb_inp2_sentences[p2][ws1] 
					oG1b = tf.transpose(oG1b,[0,2,1]) #shape is MB x Num_filter x embed_dim
					oG2b = tf.transpose(oG2b,[0,2,1])
					oG1b_normalised = self.normalise(oG1b)
					oG2b_normalised = self.normalise(oG2b)
					displacement2 = tf.sqrt(tf.reduce_sum(tf.square(tf.clip_by_value(tf.subtract(oG1b,oG2b),1e-7,1e+10)), -1))	#  MB x Num_filter
					cd2 = tf.squeeze(cosine_distance(oG1b_normalised, oG2b_normalised, dim=-1), axis=[-1])  #  MB x Num_filter
					l2diff2 = tf.sqrt(tf.reduce_sum(tf.square(tf.clip_by_value(tf.subtract(oG1b,oG2b),1e-7,1e+10)), axis=-1))	#  MB x Num_filter
					feab.append(tf.concat([cd2, l2diff2,displacement2],axis=-1))


			feaa_tensor = tf.stack(feaa)  
			feaa_tensor = tf.transpose(tf.stack(feaa),[1,0,2])  # MB x (p*ws1*ws2) x 3
			feaa_tensor = tf.reshape(feaa_tensor,[tf.shape(feaa_tensor)[0],-1]) # MB x (p*ws1*ws2*3)

			feab_tensor = tf.stack(feab)  #(p2*ws1)  x MB x (3*Num_filter)
			feab_tensor = tf.transpose(tf.stack(feab),[1,0,2])  #MB x (p2*ws1) x (3*Num_filter)
			feab_tensor = tf.reshape(feab_tensor,[tf.shape(feab_tensor)[0],-1])  #MB x (p2*ws1*3*Num_filter)
		
		return feaa_tensor,feab_tensor




	def normalise(self, a):
		with tf.name_scope("normalise"):
			norm_of_a = tf.norm(a, axis=-1) # shape: [batch_size, num_filters]
			norm_of_a = tf.expand_dims(norm_of_a,-1) # shape: [batch_size, num_filters, 1]
		return tf.divide(a, norm_of_a)


def cosine_distance(labels, predictions, dim=None):

	predictions = tf.to_float(predictions)
	labels = tf.to_float(labels)
	# predictions.get_shape().assert_is_compatible_with(labels.get_shape())
	radial_diffs = tf.multiply(predictions, labels)
	losses = 1 - tf.reduce_sum(radial_diffs, axis=(dim,), keep_dims=True)
	return losses