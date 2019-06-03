#!\usr\bin\env python
# -*- coding: utf-8 -*-
'''
Author: Eajack
date:2019/6/1
Function：
	NLP-Competition-DesignPattern
	2- 词向量训练
	(1) 学术界
		- Word2vec（已测试）
		- charVector（字向量，Word2vec）（已测试）
		- GloVe（已测试）
		- FastText（已测试）
		- ELMo（未测试，需要tensorflow-gpu，Linux）
		- BERT（未测试，基于bert-as-service，server：tensorflow-gpu，Linux）
	(2) 工业界
		- 腾讯词向量（过大未测试，理论上没问题）
		- 搜狗词向量（过大未测试，理论上没问题）
'''

import numpy as np
import logging, os, copy
import multiprocessing

from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from data.WordVectors.elmo import elmo_helpers
from bert_serving.client import BertClient

import config

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class train_wordEmbedded(object):
	"""docstring for train_wordEmbedded"""
	def __init__(self, WV_name):
		if(config.wordEmbeddedData_process_path == None):
			logging.error('train_wordEmbedded ERROR: config.wordEmbeddedData_process_path == None')
			exit(1)

		self.WV_name = WV_name
		self.inPath = config.wordEmbeddedData_process_path
		self.outPath = config.WordVector_path
		self.input_list = []
		with open(self.inPath, 'r', encoding='utf-8') as file:
			for line in file:
				line = line.strip().split()
				self.input_list.append(line)

		if(self.WV_name == 'word2vec'):
			if(not os.path.exists(os.path.join(config.WV_word2vec_path, 'word2vec_vector'))):
				self.train_word2vec()
			else:
				logging.info('train_wordEmbedded INFO: {} has been finished in path'.format(self.WV_name))
		elif(self.WV_name == 'charVector'):
			if(not os.path.exists(os.path.join(config.WV_charVector_path, 'charVector_vector'))):
				self.train_charVector()
			else:
				logging.info('train_wordEmbedded INFO: {} has been finished in path'.format(self.WV_name))
		elif(self.WV_name == 'glove'):
			self.train_glove()
		elif(self.WV_name == 'fasttext'):
			if(not os.path.exists(os.path.join(config.WV_fastext_path, 'fasttext_model'))):
				self.train_fasttext()
			else:
				logging.info('train_wordEmbedded INFO: {} has been finished in path'.format(self.WV_name))
		elif(self.WV_name == 'elmo'):
			self.train_elmo()
		elif(self.WV_name == 'bert'):
			self.train_bert()
		elif(self.WV_name == 'tencent_WV'):
			self.train_tencentWV()
		elif(self.WV_name == 'sogou_WV'):
			self.train_sogouWV()
		else:
			logging.error('train_wordEmbedded ERROR: no thie wordVector')
			logging.error('\t\tOptions: word2vec, charVector, glove, fasttext, elmo, bert, tencent_WV, sogou_WV')

		
	def train_word2vec(self):
		model = Word2Vec(LineSentence(self.inPath), size=300, window=15, min_count=5,
						 workers=multiprocessing.cpu_count())
		if( not os.path.exists(config.WV_word2vec_path) ):
			os.mkdir(config.WV_word2vec_path)
		model.save(os.path.join(config.WV_word2vec_path, 'word2vec_model'))
		model.wv.save_word2vec_format(os.path.join(config.WV_word2vec_path, 'word2vec_vector'), binary=False)

	def train_charVector(self):
		if( not os.path.exists(config.WV_charVector_path) ):
			os.mkdir(config.WV_charVector_path)

		charword_file = open(os.path.join(config.WV_charVector_path, 'wordEmbeddedData_pro_char.txt'), 'a', encoding='utf-8')
		with open(self.inPath, 'r', encoding='utf-8') as file:
			for line in file:
				line = line.strip().replace(' ','')
				line_char = [ ch for ch in line ]
				line_write = ' '.join(line_char)
				charword_file.write(line_write + '\n')

		#word2vec (char)
		model = Word2Vec(LineSentence(os.path.join(config.WV_charVector_path, 'wordEmbeddedData_pro_char.txt')), size=300, window=5, min_count=5,
						 workers=multiprocessing.cpu_count())
		model.save(os.path.join(config.WV_charVector_path, 'charVector_model'))
		model.wv.save_word2vec_format(os.path.join(config.WV_charVector_path, 'charVector_vector'), binary=False)		

	def train_glove(self):#已测试
		'''
			GloVe词向量教程：
				1- https://www.linzehui.me/2018/08/05/碎片知识/如何训练GloVe中文词向量/
				2- https://www.cnblogs.com/echo-cheng/p/8561171.html
			Attention：
				需要在Linux下进行
		'''
		if( os.path.exists(os.path.join(config.WV_glove_path, 'vectors.txt')) ):
			logging.info('train_glove INFO: train_glove sucess, in the path:{}'.format(config.WV_glove_path))
			logging.info('\t\tHow to use: \n\t\t\t\tfrom gensim.models import Word2Vec\n\t\t\t\t \
				model = Word2Vec.load_word2vec_format("vectors.txt", binary=False) ')
		else:
			logging.error('train_glove ERROR: had not train_glove!')
			logging.error('\t\tHow to train_glove: \n\t\t\t\t \
				1- "https://www.linzehui.me/2018/08/05/碎片知识/如何训练GloVe中文词向量/"\n\t\t\t\t \
				2- "https://www.cnblogs.com/echo-cheng/p/8561171.html".')

	def train_fasttext(self):
		'''
			代码参考来源：
				https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/FastText_Tutorial.ipynb
				https://zhuanlan.zhihu.com/p/48167933

			参数集合：
				(https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/FastText_Tutorial.ipynb)
			- model: Training architecture. Allowed values: `cbow`, `skipgram` (Default `cbow`)
			- size: Size of embeddings to be learnt (Default 100)
			- alpha: Initial learning rate (Default 0.025)
			- window: Context window size (Default 5)
			- min_count: Ignore words with number of occurrences below this (Default 5)
			- loss: Training objective. Allowed values: `ns`, `hs`, `softmax` (Default `ns`)
			- sample: Threshold for downsampling higher-frequency words (Default 0.001)
			- negative: Number of negative words to sample, for `ns` (Default 5)
			- iter: Number of epochs (Default 5)
			- sorted_vocab: Sort vocab by descending frequency (Default 1)
			- threads: Number of threads to use (Default 12)

			In addition, FastText has three additional parameters -
			- min_n: min length of char ngrams (Default 3)
			- max_n: max length of char ngrams (Default 6)
			- bucket: number of buckets used for hashing ngrams (Default 2000000)
		'''
		if( not os.path.exists(config.WV_fastext_path) ):
			os.mkdir(config.WV_fastext_path)

		fasttext_model = FastText(size=300, window=5, min_count=5, \
			iter=10, min_n = 3 , max_n = 6)
		# build the vocabulary
		fasttext_model.build_vocab(sentences=self.input_list)
		# train the model
		fasttext_model.train(sentences=self.input_list, total_examples=len(self.input_list),epochs=10)
		# save
		fasttext_model.save(os.path.join(config.WV_fastext_path, 'fasttext_model'))

	def train_elmo(self):#未测试
		'''
			ELMo词向量教程：
				1- https://www.linzehui.me/2018/08/12/碎片知识/如何将ELMo词向量用于中文/
			Attention：
				需要tensorflow-gpu，Linux下运行
		'''
		if( os.path.exists(os.path.join(config.WV_elmoOutputs_path, 'vocab_embedding.hdf5')) ):
			logging.info('train_elmo INFO: train_elmo sucess, in the path:{}'.format(config.WV_elmo_path))
			logging.info('\t\tHow to use:\n\t\t\t\t \
				1- https://www.linzehui.me/2018/08/12/碎片知识/如何将ELMo词向量用于中文/')
		else:
			logging.error('train_elmo ERROR: had not train_elmo!')
			logging.error('\t\tHow to train_elmo: \n\t\t\t\t \
				1- "https://www.linzehui.me/2018/08/12/碎片知识/如何将ELMo词向量用于中文/"')

	def train_bert(self):
		logging.info('train_bert INFO: use bert-as-service')
		logging.info('\t\tHow to use:\n\t\t\t\t1- https://zhuanlan.zhihu.com/p/50582974\
			\n\t\t\t\t2- https://github.com/hanxiao/bert-as-service')

	def train_tencentWV(self):
		if( os.path.exists(os.path.join(config.WV_tencent_path, 'Tencent_AILab_ChineseEmbedding.txt')) ):
			logging.info('train_tencentWV INFO: train_tencentWV sucess, in the path: {}'.format(config.WV_tencent_path))
			logging.info('\t\tHow to use: \n\t\t\t\tfrom gensim.models.word2vec import KeyedVectors\n\t\t\t\t \
				wv_from_text = KeyedVectors.load_word2vec_format(file, binary=False) ')
		else:
			logging.error('train_tencentWV ERROR: had not train_tencentWV!')
			logging.error('\t\tHow to get: \n\t\t\t\t \
				1- "https://ai.tencent.com/ailab/nlp/embedding.html"')

	def train_sogouWV(self):
		if( os.path.exists(os.path.join(config.WV_sogou_path, 'sgns.sogou.word')) ):
			logging.info('train_sogouWV INFO: train_sogouWV sucess, in the path: {}'.format(config.WV_sogou_path))
			logging.info('\t\tHow to use: \n\t\t\t\tfrom gensim.models.word2vec import KeyedVectors\n\t\t\t\t \
				wv_from_text = KeyedVectors.load_word2vec_format(file, binary=False) ')
		else:
			logging.error('train_sogouWV ERROR: had not train_sogouWV!')
			logging.error('\t\tHow to get: \n\t\t\t\t \
				1- "https://www.flyai.com/m/sgns.sogou.word.zip"')


class get_wordEmbedded(object):
	"""docstring for train_wordEmbedded"""
	def __init__(self, WV_name, text_list, max_word_length):
		'''
			e.g. WV_name = word2vec
				 text_list = [
								'想当年 佘山 没有 三品 香算镇 起来 像样 饭店 菜品 ',
								'想当年 佘山 没有',
								'想当年 佘山 没有 三品 香算镇 起来 像样'
							]
				max_word_length = 300(一句平均最多300个词/字)
		'''
		static_WV = ['word2vec', 'charVector', 'glove', 'fasttext', 'tencent_WV', 'sogou_WV']
		dynamic_WV = ['elmo', 'bert']
		self.WV_name = WV_name
		self.max_word_length = max_word_length

		#cut words, limit word num: max_word_length
		text_list_buffer = copy.deepcopy(text_list)
		self.text_list = []
		
		for text in text_list_buffer:
			text_buffer, text_split = [], []
			if(self.WV_name != 'charVector'):
				text_split = text.strip().split()
			else:
				text_split = text.strip().replace(' ','')
				text_split = [ ch for ch in text_split ]

			if(len(text_split) > self.max_word_length):
				text_buffer = copy.deepcopy(text_split[0:self.max_word_length])
			text_buffer = copy.deepcopy(text_split)
			(self.text_list).append(text_buffer)

		# path
		self.WV_path = ''
		self.wordVectors = 0
		self.wv = 0
		if(self.WV_name in static_WV):
			if(self.WV_name == 'word2vec'):
				self.WV_path = os.path.join(config.WV_word2vec_path, 'word2vec_vector')
			elif(self.WV_name == 'charVector'):
				self.WV_path = os.path.join(config.WV_charVector_path, 'charVector_vector')
			elif(self.WV_name == 'glove'):
				self.WV_path = os.path.join(config.WV_glove_path, 'vectors.txt')
			elif(self.WV_name == 'fasttext'):
				self.WV_path = os.path.join(config.WV_fastext_path, 'fasttext_model')
			elif(self.WV_name == 'tencent_WV'):
				self.WV_path = os.path.join(config.WV_tencent_path, 'Tencent_AILab_ChineseEmbedding.txt')
			elif(self.WV_name == 'sogou_WV'):
				self.WV_path = os.path.join(config.WV_sogou_path, 'sgns.sogou.word')
			else:
				pass

			if(self.WV_name != 'fasttext'):
				self.wordVectors = KeyedVectors.load_word2vec_format(self.WV_path, binary=False)
			else:
				self.wordVectors = FastText.load(self.WV_path)
			self.wv = self.get_stacitWV()

		elif(self.WV_name in dynamic_WV):
			if(self.WV_name == 'elmo'):
				self.wv = self.get_elmo()
			elif(self.WV_name == 'bert'):
				self.wv = self.get_bert()
			else:
				pass

		else:
			logging.error('get_wordEmbedded ERROR: no thie wordVector')
			logging.error('\t\tOptions: word2vec, charVector, glove, fasttext, elmo, bert, tencent_WV, sogou_WV')		


	def get_stacitWV(self):
		batch_size = len(self.text_list)
		wv_all, wv_dim = 0, 0
		if(self.WV_name != 'tencent_WV'):
			wv_dim = 300
		else:
			wv_dim = 200

		wv_all = np.zeros((batch_size, self.max_word_length, wv_dim))
		for bathc_cnt, text in enumerate(self.text_list):
			if(len(text) > self.max_word_length):
				logging.error('len(text) = {} > self.max_word_length = {}'.format(len(text), self.max_word_length))
				return None
			elif(len(text) <= self.max_word_length):
				# 小于self.max_word_length，初始化已补0
				text_wv = np.zeros((self.max_word_length, wv_dim))
				wv_now = np.zeros((1, wv_dim))
				for word_cnt, word in enumerate(text):
					try:
						wv_now = self.wordVectors[word]
					except KeyError:
						try:
							wv_now = self.wordVectors['<unk>']
						except KeyError:
							pass
					text_wv[word_cnt] = wv_now
				wv_all[bathc_cnt] = text_wv

		return wv_all

	def get_elmo(self):
		if( not os.path.exists(os.path.join(config.WV_elmoOutputs_path, 'vocab_embedding.hdf5')) ):
			logging.error('get_elmo ERROR: had not train_elmo!')
			logging.error('\t\tHow to train_elmo: \n\t\t\t\t \
				1- "https://www.linzehui.me/2018/08/12/碎片知识/如何将ELMo词向量用于中文/"')

		wv_all = []
		# 按batch_size = 128 feed
		batch_size_elmo = 128
		data_size = len(self.text_list)
		batch_num = 0
		batch_num_int = data_size//batch_size_elmo
		batch_num_float = data_size/batch_size_elmo
		if(batch_num_float-batch_num_int==0):
			batch_num = batch_num_int
		else:
			batch_num = batch_num_int + 1

		for batch_count in range(batch_num):
			begin_index = batch_count*batch_size_elmo
			end_index = min((batch_count + 1) * batch_size_elmo, data_size)
			content_128 = self.text_list[begin_index:end_index]
			contend_128_np = elmo_helpers.get_ELMo(content_128)

			# 强制contend_128_np补0
			# because 可能batch中最长句子词数 < self.max_word_length
			# 技巧：np.concatenate补0
			if(contend_128_np.shape[1] < self.max_word_length):
				padding_zeros = np.zeros([contend_128_np.shape[0], \
					self.max_word_length-contend_128_np.shape[1], contend_128_np.shape[2]])
				contend_128_np = np.concatenate( (contend_128_np, padding_zeros),axis=1 )

			if(batch_count == 0):
				wv_all = contend_128_np
			else:
				wv_all = np.concatenate( (wv_all,contend_128_np),axis=0 )

		return wv_all

	def get_bert(self):
		bc = BertClient(ip='xx.xx.xx.xx', port=5555)  # ip address of the GPU machine
		return bc.encode(self.text_list)


if __name__ == '__main__':
	#test
	input_text = [
		'想当年 佘山 没有 三品 香算镇 起来 像样 饭店 菜品 ',
		'想当年 佘山 没有',
		'想当年 佘山 没有 三品 香算镇 起来 像样'
	]
	wv_list = ['word2vec','charVector','glove','fasttext','tencent_WV','sogou_WV','elmo','bert']
	#1- train & get
	for wv in wv_list:
		train_test = train_wordEmbedded(wv)
		get_text = get_wordEmbedded(wv, input_text, 300)
		trash = 1
