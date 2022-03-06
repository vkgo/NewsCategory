import torch
import jieba
import objprocess as op
from torch import nn
from torchtext.vocab import Vectors
from torchtext.legacy.data import Field
from Module import TextCNN
from loaddataset import jiebatokenize

stoi = op.load_obj('TEXT_vocab_stoi')
itos = op.load_obj('LABEL_vocab_itos')


print(len(stoi))
wordvectors = Vectors(name='../data/sgns.sogounews.bigram-char_3')
print(len(wordvectors))

temp = input('新闻:')
index = stoi.get(temp)
print('index:{}'.format(index))



wordvectors = Vectors(name='../data/sgns.sogounews.bigram-char_3')
print(wordvectors.itos[index])