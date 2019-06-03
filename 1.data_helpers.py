#!\usr\bin\env python
# -*- coding: utf-8 -*-
'''
Author: Eajack
date:2019/6/1
Function：
	NLP-Competition-DesignPattern
	1- 数据预处理（暂无加速）
		标点符号 => emoji => 连句号处理 => 分句 => 分词
'''

import logging
import pandas as pd
import numpy as np
import re,time,copy,os,shutil
import pyltp
from pyltp import Segmentor, Postagger

import config

def get_wordEmbeddedData(textColumnName):
	pd_train = pd.read_csv(config.trainData_path, encoding='utf-8')
	pd_valid = pd.read_csv(config.validData_path, encoding='utf-8')

	if( os.path.exists(config.externalData_path) ):
		shutil.copyfile(config.externalData_path, config.wordEmbeddedData_path)

	text_list, valid_text = pd_train[textColumnName].tolist(), pd_valid[textColumnName].tolist()
	text_list.extend(valid_text)
	with open(config.wordEmbeddedData_path, 'a', encoding='utf-8') as file:
		for text in text_list:
			text = str(text)
			text = re.sub(re.compile(r'\s+'), ' ', text)
			file.write(text+'\n')

def load_stopwords():
	if(config.stopwords_path == None):
		logging.error('load_stopwords ERROR: config.stopwords_path == None')
		exit(1)

	stopwords = []
	with open(config.stopwords_path, 'r', encoding='utf-8') as file:
		for line in file:
			line = line.strip()
			stopwords.append(line)

	return stopwords

def _is_chinese_char(ch):
	# 参考：谷歌BERT模型
	"""Checks whether CP is the codepoint of a CJK character."""
	# This defines a "chinese character" as anything in the CJK Unicode block:
	#   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
	#
	# Note that the CJK Unicode block is NOT all Japanese and Korean characters,
	# despite its name. The modern Korean Hangul alphabet is a different block,
	# as is Japanese Hiragana and Katakana. Those alphabets are used to write
	# space-separated words, so they are not treated specially and handled
	# like the all of the other languages.
	cp = ord(ch)
	if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
		(cp >= 0x3400 and cp <= 0x4DBF) or  #
		(cp >= 0x20000 and cp <= 0x2A6DF) or  #
		(cp >= 0x2A700 and cp <= 0x2B73F) or  #
		(cp >= 0x2B740 and cp <= 0x2B81F) or  #
		(cp >= 0x2B820 and cp <= 0x2CEAF) or
		(cp >= 0xF900 and cp <= 0xFAFF) or  #
		(cp >= 0x2F800 and cp <= 0x2FA1F)):  #
	  return True

	return False

def process_puncuation(content):
	content_punPro = copy.deepcopy(content)
	#(1)- 空白字符 => 中文句号
	content_punPro = re.sub(re.compile(r'\s+'), '。', content_punPro)

	#(2)- 英文标点符号 => 中文句号
	eng_puncuations = '[<>[\]{}!@#$%\^&*()_\-=+\\|;:\'",./?`~*]'
	content_punPro = re.sub(eng_puncuations, '。', content_punPro)

	#(3)- 所有中文标点符号 => 中文句号
	chi_puncations = '[·~！@#￥%……&*（）——+\-={}【】、|；：‘’“”，《》/？*～]'
	content_punPro = re.sub(chi_puncations, '。', content_punPro)

	return content_punPro

def deleteEmoji(content):
	#参考
	## 1- https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py
	## 2- https://zhuanlan.zhihu.com/p/41213713
	## 3- https://apps.timwhitlock.info/emoji/tables/unicode#block-6c-other-additional-symbols
	content_noEmoji = copy.deepcopy(content)
	try:
		# Wide UCS-4 build
		## 是这个
		myre = re.compile(u'['
						  u'\U0001F300-\U0001F64F'
						  u'\U0001F680-\U0001F6FF'
						  u'\u2600-\u2B55'
						  u'\u23cf'
						  u'\u23e9'
						  u'\u231a'
						  u'\u3030'
						  u'\ufe0f'
						  u"\U0001F600-\U0001F64F"  # emoticons
						   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
							u'\U00010000-\U0010ffff'
						   u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
						   u'\U00002702-\U000027B0]+',     
						  re.UNICODE)
	except re.error:
		# Narrow UCS-2 build
		myre =   re.compile(u'('
								  u'\ud83c[\udf00-\udfff]|'
								  u'\ud83d[\udc00-\ude4f]|'
								  u'\uD83D[\uDE80-\uDEFF]|'
								  u"(\ud83d[\ude00-\ude4f])|"  # emoticon
								  u'[\u2600-\u2B55]|'
								  u'[\u23cf]|'
								  u'[\u1f918]|'
									u'[\u23e9]|'
								  u'[\u231a]|'
								  u'[\u3030]|'
								  u'[\ufe0f]|'
								  u'\uD83D[\uDE00-\uDE4F]|'
								  u'\uD83C[\uDDE0-\uDDFF]|'
								u'[\u2702-\u27B0]|'
								  u'\uD83D[\uDC00-\uDDFF])+',
								  re.UNICODE)
	content_noEmoji=myre.sub('。', content_noEmoji)
	return content_noEmoji

def divideSentence(content):
	#分句后，每个句子中只允许有中文、英文、数字
	content_after_DS = ''
	for sentence in content.split('。'):
		sentence_afterFilter = ''
		for ch in sentence:
			if(_is_chinese_char(ch) or ch.isalpha() or ch.isdigit()):
				sentence_afterFilter += ch
		content_after_DS += (sentence_afterFilter+' ')

	return content_after_DS

def get_wordsList(content_after_DS, stopwords_flag=False):
	stopwords = []
	if(stopwords_flag):
		stopwords = load_stopwords()

	wordSegmentor_ltp = Segmentor()
	cws_model_path = config.LTP_DATA_DIR + r'\cws.model'
	wordSegmentor_ltp.load(cws_model_path)

	words_list = []
	for text in content_after_DS.split():
		words_segment_list = list(wordSegmentor_ltp.segment(text))
		words_list.extend(words_segment_list)
		if(stopwords_flag):
			words_list = [ word for word in words_list if(word not in stopwords) ]

	wordSegmentor_ltp.release()
	words_list_str = ' '.join(words_list)

	return words_list_str

def data_process(content):
	content_pro = copy.deepcopy(content.strip())

	#1- 标点符号处理
	content_pro = process_puncuation(content_pro)

	#2- emoji => 中文句号
	content_pro = deleteEmoji(content_pro)

	#3- 连句号处理
	content_pro = re.sub(r'。{2,}', '。', content_pro) #对多个句号转换为单句号

	#4- 分句
	content_pro = divideSentence(content_pro)
	content_pro = content_pro.strip()

	#5- 分词
	content_pro = get_wordsList(content_pro)

	return content_pro

def data_process_txt(input_txtName, output_txtName):
	writeFile = open(output_txtName, 'a', encoding='utf-8')
	with open(input_txtName, 'r', encoding='utf-8') as file:
		for line in file:
			line = line.strip()
			content_pro = data_process(line)
			writeFile.write(content_pro+'\n')

def data_process_csv(input_csvName, output_csvName, textColumnName):
	pd_train = pd.read_csv(input_csvName, encoding='utf-8')
	pd_train_copy = copy.deepcopy(pd_train)
	
	for row_cnt in range(0,len(pd_train_copy)):
		temp = str(pd_train_copy.iloc[row_cnt][textColumnName])
		content_pro = data_process(temp)
		pd_train_copy.loc[row_cnt, textColumnName] = content_pro
		print(row_cnt)

	pd_train_copy.to_csv(output_csvName, index=False, encoding='utf-8')


if __name__ == '__main__':
	# functions test
	get_wordEmbeddedData('comment')

	data_process_txt(config.wordEmbeddedData_path, config.wordEmbeddedData_process_path)
	data_process_csv(config.trainData_path, config.trainData_process_path, 'comment')
	data_process_csv(config.validData_path, config.validData_process_path, 'comment')
