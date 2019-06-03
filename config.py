#!\usr\bin\env python
# -*- coding: utf-8 -*-

import os

project_path = r'F:\STUDYING\MyProjects\pycharm\NLP-Competition-DesignPattern'#项目绝对路径
LTP_DATA_DIR = os.path.join(project_path, 'data', 'LTP')
extraCodes_path = os.path.join(project_path, 'data', 'Extra_Codes')

# 输入
input_path = os.path.join(project_path, 'data', 'input')
stopwords_path = os.path.join(input_path, 'stopwords.txt')#个人准备1

trainData_path = os.path.join(input_path, 'train.csv')#比赛提供1
validData_path = os.path.join(input_path, 'valid.csv')#比赛提供2
testData_path = os.path.join(input_path, 'test.csv')#比赛提供3
externalData_path = os.path.join(input_path, 'externalData.txt')#个人准备2
wordEmbeddedData_path = os.path.join(input_path, 'wordEmbeddedData.txt')

trainData_process_path = os.path.join(input_path, 'train_pro.csv')
validData_process_path = os.path.join(input_path, 'valid_pro.csv')
testData_process_path = os.path.join(input_path, 'test_pro.csv')
wordEmbeddedData_process_path = os.path.join(input_path, 'wordEmbeddedData_pro.txt')

WordVector_path = os.path.join(project_path, 'data', 'WordVectors')
WV_word2vec_path = os.path.join(WordVector_path, 'word2vec')
WV_charVector_path = os.path.join(WordVector_path, 'charVector')
WV_glove_path = os.path.join(WordVector_path, 'glove')
WV_fastext_path = os.path.join(WordVector_path, 'fasttext')
WV_elmo_path = os.path.join(WordVector_path, 'elmo')
WV_elmoFiles_path = os.path.join(WV_elmo_path, 'elmo_files')
WV_elmoOutputs_path = os.path.join(WV_elmo_path, 'elmo_files', 'elmo_outputs')
WV_bert_path = os.path.join(WordVector_path, 'bert')
WV_tencent_path = os.path.join(WordVector_path, 'tencent_WV')
WV_sogou_path = os.path.join(WordVector_path, 'sogou_WV')

# 输出
output_path = os.path.join(project_path, 'data', 'output')