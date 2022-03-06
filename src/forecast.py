import torch
from torch import nn

import objprocess as op
from loaddataset import jiebatokenize

class forecast:
    def __init__(self, module_loc="../SavedModel/mymodule_31.pth", usegpu=1):
        # 加载vocab
        self.TEXT_vocab_stoi = op.load_obj('TEXT_vocab_stoi')
        self.TEXT_vocab_itos = op.load_obj('TEXT_vocab_itos')
        self.LABEL_vocab_itos = op.load_obj('LABEL_vocab_itos')
        self.vocab_length = len(self.TEXT_vocab_stoi)



        # 加载模型
        if usegpu == 1:
            if torch.cuda.is_available():
                self.usegpu = usegpu
                self.module = torch.load(module_loc, map_location=lambda storage, loc: storage.cuda())
            else:
                self.usegpu = 0
                self.module = torch.load(module_loc)
        else:
            self.module = torch.load(module_loc)

        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()


    def get(self, rawnews):
        # 分割
        wordlist = jiebatokenize(rawnews)

        # 转码
        embedvec = []
        for item in wordlist:
            index = self.TEXT_vocab_stoi.get(item)
            if index != None:
                if index < self.vocab_length:
                    embedvec.append(index)

        embedvec = torch.Tensor(embedvec)
        embedvec = embedvec.unsqueeze(1)

        self.module.eval()

        if self.usegpu == 1:
            self.loss_fn = self.loss_fn.cuda()
            self.module = self.module.cuda()
            embedvec = embedvec.cuda()

        module_output = self.module(embedvec.long())
        label_index = module_output.argmax(1).item()

        return self.LABEL_vocab_itos[label_index]

