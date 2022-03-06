import torch
from torch import nn



class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(config.vocabulary_length, config.embedding_dimension) # 定义embedding层
        self.embedding = self.embedding.from_pretrained(config.wordvectors)  # 加载词向量


        # 卷积层
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=config.filternumber, kernel_size=[n, config.embedding_dimension]) for n in config.filterlength]
            # 卷积核尺寸 eg:[128, 128, 3, 300]
        )

        # ReLu
        self.relu = nn.ReLU()

        # dropout
        self.dropout = nn.Dropout(config.dropoutp)  # dropout

        # fullconnect
        self.fc = nn.Linear(len(config.filterlength) * config.filternumber, config.classesnumber)  # 全连接层

    def maxpool(self, x, length):
        mpool = nn.MaxPool1d(kernel_size=length)
        return mpool(x)


    def forward(self, x):

        # 2022-2-17报错:
        # Expected 4-dimensional input for 4-dimensional weight [128, 1, 3, 300],
        # but got 3-dimensional input of size [1853, 128, 300] instead
        # 卷积尺寸对不上。
        # 初始x torch.Size([1769, 128])

        x = self.embedding(x) # [1853, 128, 300]

        # 经过embedding后 torch.Size([1769, 128, 300])
        x = x.unsqueeze(0) # [1, 1853, 128, 300]
        # torch.Size([1, 1769, 128, 300])

        x = x.permute(2, 0, 1, 3) # [128, 1, 1853, 300]
        # torch.Size([128, 1, 1769, 300])

        x = [conv2d(x) for conv2d in self.convs] # len(x) = len(config.filterlength)
        # [[128, 128, xx, 1]
        #  [128, 128, xx - 1, 1]
        #  [128, 128, xx - 2, 1]]

        x = [tempx.squeeze(3) for tempx in x] # 将最后一维去除(embedding_dimension)
        # [[128, 128, xx]
        #  [128, 128, xx - 1]
        #  [128, 128, xx - 2]]

        x = [self.relu(tempx) for tempx in x]

        x = [self.maxpool(tempx, tempx.size(2)) for tempx in x]
        # [128, 128, 1]
        # [128, 128, 1]
        # [128, 128, 1]

        x = torch.cat(x, dim=1) # [128, 384, 1]
        # torch.Size([128, 384, 1])

        x = x.squeeze(2)
        # IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
        # torch.Size([128, 384])

        x = self.dropout(x)

        # 2022-2-18报错
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (49152x1 and 384x11)
        #                                                       128*3*128*1 384*11
        # torch.Size([128, 384, 1])
        x = self.fc(x)
        return x

