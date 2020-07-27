pytorch处理文本数据代码版本1-处理文本相似度数据


下面的代码，相比于版本2的代码，并没有使用gensim，而且处理的时候针对的是每一个样本，也就是每一行，也就是
sentence1和sentence2并没有拆开来处理。

整体代码是我自己完全整理出来的，比较整齐

```python

"""
@author: DASOU
@time: 20200726
"""
import torch
import os
import pickle as pkl

## 读取原始数据，生成对应的word2index
def get_word_voc(config_base):
    train_path=config_base.train_path
    file=open(train_path,'r')
    lines=file.readlines()
    min_freq,max_size,UNK,PAD=config_base.min_freq,config_base.max_size,config_base.UNK,config_base.PAD
    vocab_dic={}
    for line in lines:
        try:
            line=line.strip().split('\t')
        except:
            print('The data formate is not correct,please correct it as example data')
            exit()
        try:
            if len(line)==3:
                sen=line[0]+line[1]
                tokenizer = lambda x: [y for y in x]
                for word in tokenizer(sen):
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1 ## 为了计算出每个单词的词频，为之后过滤低频词汇做准备
        except:
            print('The data formate is not correct,please correct it as example data')
            exit()
    file.close()
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]## 是为了计算每个单词的词频
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}## 过滤掉低频词汇之后我们按照顺序来word-index的映射
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1}) ## 补充unkonw和pad字符对应的数字
    return vocab_dic


def load_data(cate,vocab_dic,config_base):
    if cate=='train':
        data_path=config_base.train_path
    elif cate=='dev':
        data_path = config_base.dev_path
    else:
        data_path = config_base.test_path
    file=open(data_path,'r')
    contents=[]
    for line in file.readlines():
        words_line1=[]
        words_line2=[]
        line=line.strip().split('\t')
        sen1,sen2,label=line[0],line[1],line[2]
        tokenizer = lambda x: [y for y in x]
        token_sen1=tokenizer(sen1)
        token_sen2 = tokenizer(sen2)
        sen1_len = len(token_sen1)
        sen2_len = len(token_sen2)

        if config_base.pad_size:
            if len(token_sen1) < config_base.pad_size:
                token_sen1.extend([config_base.PAD] * (config_base.pad_size - len(token_sen1)))
            else:
                token_sen1 = token_sen1[:config_base.pad_size]

            if len(token_sen2) < config_base.pad_size:
                token_sen2.extend([config_base.PAD] * (config_base.pad_size - len(token_sen2)))
            else:
                token_sen2 = token_sen2[:config_base.pad_size]
        for word1 in token_sen1:
            words_line1.append(vocab_dic.get(word1, vocab_dic.get(config_base.UNK)))

        for word2 in token_sen2:
            words_line2.append(vocab_dic.get(word2, vocab_dic.get(config_base.UNK)))
        contents.append((words_line1,words_line2,int(label)))
    return contents

# 导入/训练对应的word2index
def get_w2i(config_base):

    if not os.path.exists(config_base.w2i_path):
        print('There is not a pre word2index,now is to process data for geting word2index')
        vocab_dic = get_word_voc(config_base)
        pkl.dump(vocab_dic, open(config_base.w2i_path, 'wb'))
        vord_size = len(vocab_dic)
    else:
        print('There is pre word2index, now is to load the pre infomation')
        vocab_dic = pkl.load(open(config_base.w2i_path, 'rb'), encoding='utf-8')
        vord_size = len(vocab_dic)
    return vocab_dic,vord_size

class DatasetIterater():
    def __init__(self, batches, config_base):
        self.batch_size = config_base.batch_size
        self.batches = batches
        self.n_batches = len(batches) // config_base.batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = config_base.device

    def _to_tensor(self, datas):
        x1 = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        x2 = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        return (x1, x2), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset,config_base):
    iter = DatasetIterater(dataset,config_base)
    return iter

```