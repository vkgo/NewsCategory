import torch
import jieba
import objprocess as op
from torch import nn
from torchtext.vocab import Vectors
from torchtext.legacy.data import Field
from Module import TextCNN
from loaddataset import jiebatokenize



# 加载词向量
# wordvectors = Vectors(name='../data/sgns.sogounews.bigram-char_3')
# print(wordvectors)
TEXT_vocab_stoi = op.load_obj('TEXT_vocab_stoi')
TEXT_vocab_itos = op.load_obj('TEXT_vocab_itos')
LABEL_vocab_itos = op.load_obj('LABEL_vocab_itos')
vocab_length = len(TEXT_vocab_stoi)
print(LABEL_vocab_itos)

news = input("输入新闻:")
wordlist = jiebatokenize(news)
print(wordlist)
print(len(wordlist))

# stoi = wordvectors.stoi
stoi = TEXT_vocab_stoi
# embedvec = [stoi.get(item) for item in wordlist]
# print(embedvec)
# embedvec.clear()
embedvec = []
for item in wordlist:
    index = stoi.get(item)
    if index != None:
        if index < vocab_length:
            embedvec.append(index)
            print(TEXT_vocab_itos[index])

embedvec = torch.Tensor(embedvec)
embedvec = embedvec.unsqueeze(1)
print(embedvec)
print(embedvec.shape)
module = torch.load("../SavedModel/mymodule_24.pth", map_location=lambda storage, loc: storage.cuda())
module.eval()
print(module.parameters)

loss_fn = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
    module = module.cuda()
    embedvec = embedvec.cuda()

module_output = module(embedvec.long())
print(module_output)
label_index = module_output.argmax(1).item()
print(label_index)
print(LABEL_vocab_itos[label_index])
