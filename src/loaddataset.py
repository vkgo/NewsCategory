import jieba
import torch
import config as c
import objprocess as op
from torchtext.legacy.data import Field, TabularDataset
from torchtext.vocab import Vectors

# 返回分割后剩下的词数据
# 去除了停用词



def jiebatokenize(sentence):
    # 获取停用词
    with open('../data/hit_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read().split('\n')
    return [token for token in jieba.lcut(sentence) if token not in stopwords]


def load():
    # 创建Field对象
    TEXT = Field(sequential=True, tokenize=jiebatokenize, lower=True)
    LABEL = Field(sequential=False, dtype=torch.int64)

    wordvectors = Vectors(name='../data/sgns.sogounews.bigram-char_3')


    # 读取数据
    traindataset, valdataset, testdataset = TabularDataset.splits(
        path='../data',
        train='cnews.train.txt',
        validation='cnews.val.txt',
        test='cnews.test.txt',
        format='tsv',
        fields=[('Label', LABEL), ('Text', TEXT)]
    )
    
    #读取词向量


    #建立词表
    TEXT.build_vocab(traindataset, valdataset, testdataset, vectors=wordvectors) #
    # TEXT.build_vocab()
    # TEXT.build_vocab()
    # TEXT.vocab.load_vectors(vectors=wordvectors)

    LABEL.build_vocab(traindataset, valdataset, testdataset)

    op.save_obj(TEXT.vocab.stoi, name='TEXT_vocab_stoi')
    op.save_obj(TEXT.vocab.itos, name='TEXT_vocab_itos')
    op.save_obj(LABEL.vocab.itos, name='LABEL_vocab_itos')


    config = c.Config(vocabulary_length=len(TEXT.vocab),
                      embedding_dimension=TEXT.vocab.vectors.size()[-1],
                      wordvectors=TEXT.vocab.vectors,
                      classesnumber=len(LABEL.vocab)
                      )

    return traindataset, valdataset, testdataset, config




