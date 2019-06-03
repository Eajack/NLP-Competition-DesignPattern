'''
!/usr/bin/env python3
-*- coding : utf-8 -*-
Author: Eajack
date:2019/3/19 - 
Function：
	暨大毕设-刘译键
	1- ELMo词向量生成
		参考：https://www.linzehui.me/2018/08/12/碎片知识/如何将ELMo词向量用于中文/
'''
import tensorflow as tf
import os, config
from data.WordVectors.elmo.elmo_files import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
	dump_token_embeddings
tf.reset_default_graph()

def get_ELMo(input_text):
	elmo_context_input_ = 0
	graph1 = tf.Graph()
	with graph1.as_default():
		tokenized_context = [sentence.split() for sentence in input_text]
		tokenized_question = [
			['这', '是', '什么'],
		]

		vocab_file = config.elmo_vocab_path
		options_file = config.elmo_options_path
		weight_file = config.elmo_weight_path
		token_embedding_file = config.elmo_tokenEmbedding_path

		## Now we can do inference.
		# Create a TokenBatcher to map text to token ids.
		batcher = TokenBatcher(vocab_file)

		# Input placeholders to the biLM.
		context_token_ids = tf.placeholder('int32', shape=(None, None))
		question_token_ids = tf.placeholder('int32', shape=(None, None))

		# Build the biLM graph.
		bilm = BidirectionalLanguageModel(
			options_file,
			weight_file,
			use_character_inputs=False,
			embedding_weight_file=token_embedding_file
		)

		# Get ops to compute the LM embeddings.
		context_embeddings_op = bilm(context_token_ids)
		question_embeddings_op = bilm(question_token_ids)

		elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
		with tf.variable_scope('', reuse=True):
			# the reuse=True scope reuses weights from the context for the question
			elmo_question_input = weight_layers(
				'input', question_embeddings_op, l2_coef=0.0
			)

		elmo_context_output = weight_layers(
			'output', context_embeddings_op, l2_coef=0.0
		)
		with tf.variable_scope('', reuse=True):
			# the reuse=True scope reuses weights from the context for the question
			elmo_question_output = weight_layers(
				'output', question_embeddings_op, l2_coef=0.0
			)

		with tf.Session() as sess:
			# It is necessary to initialize variables once before running inference.
			sess.run(tf.global_variables_initializer())

			# Create batches of data.
			context_ids = batcher.batch_sentences(tokenized_context)
			question_ids = batcher.batch_sentences(tokenized_question)

			# Compute ELMo representations (here for the input only, for simplicity).
			elmo_context_input_, elmo_question_input_ = sess.run(
				[elmo_context_input['weighted_op'], elmo_question_input['weighted_op']],
				feed_dict={context_token_ids: context_ids,
						   question_token_ids: question_ids}
			)

		#print(input_text, elmo_context_input_)
	return elmo_context_input_

if __name__ == '__main__':
	# Our small dataset.
	input_text = [
		'想当年 佘山 没有 三品 香算镇 起来 像样 饭店 菜品 ',
		'想当年 佘山 没有',
		'想当年 佘山 没有 三品 香算镇 起来 像样'
	]
	input_elmo = get_ELMo(input_text)
	print(input_elmo)