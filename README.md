# NLP-Competition-DesignPattern

## updated on 2019/6/3

## 1、运行环境
> * Windows or Linux
> * Python3.5.2(Python 3.x.x)

## 2、第三方库汇总
>* pip3 install -r requirements.txt

## 3、项目用途
个人整合在NLP类竞赛中通用代码，e.g. 文本预处理 etc. ，方便比赛时候直接调用

## 4、代码&文件夹说明
PS：由于GitHub不允许上传 >= 100MB文件，因此很多中间结果、结果文档删了，只保留文件夹

1. `config.py`：项目公用路径

2. `1.data_helpers.py`：文本数据处理常用函数
	* `get_wordEmbeddedData(textColumnName)`：统计训练集`train.csv`、验证集`valid.csv` & 外部文本集`externalData.txt`（若有）的所有文本数据，`textColumnName`为csv文本对应列标签。用于DL模型构建词向量。
	* `data_process_txt(input_txtName, output_txtName)`：对txt文本预处理，预处理函数为`data_process(content)`，包括标点符号处理 => emoji处理 => 连句号处理 => 分句处理 => 分词处理。`input_txtName`为输入txt文档路径，`output_txtName`为输出txt文档路径。**txt文档格式：1条文本/行**
	* `data_process_csv(input_csvName, output_csvName, textColumnName)`：对csv文本预处理，预处理函数为`data_process(content)`。`input_csvName`为输入csv文档路径，`output_csvName`为输出csv文档路径，`textColumnName`为csv文本对应列标签。

3. `2.wordVectors_helpers`：预处理后获得`wordEmbeddedData_pro.txt`，利用该文本训练词向量。**目前支持词向量：Word2vec、charVector（字向量，Word2vec版）、GloVe、FastText、ELMo、BERT、腾讯词向量、搜狗词向量。** 部分未测试，详见代码注释。

4. `data文件夹`：所有数据、import代码 etc.
	* input：竞赛官方给定的数据，包括`train.csv`、`valid.csv`、`test.csv`
	* output：模型输出结果
	* LTP：LTP库所需文档
	* WordVectors：`2.wordVectors_helpers`词向量生成结果 & 相关代码 etc.
	* Extra_Codes：部分词向量生成所需外部代码 & 其他外部代码，e.g. ELMo词向量生成需要bilm-tf代码
	* stop_words.txt：常用的中文NLP停用词表

## 5、More
repo保持update，后续继续补全无关NLP下游任务的东西，同时对NLP下游任务分类整理模型，e.g. 文本分类任务、文本生成任务 etc.
