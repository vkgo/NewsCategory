class Config:
    def __init__(self,
                vocabulary_length,  # 词表长度,int
                embedding_dimension,  # 词向量维度,int
                wordvectors,  # 词向量，Vectors
                classesnumber, # 最后分类的个数，int
                dropoutp = 0.5,  # dropout层概率，float
                filterlength = [3, 4, 5],  # 卷积核的长度，array
                filternumber = 64,  # 同大小的卷积核数目(Out_channel number)，int
                learningrate = 1e-3,  # 学习速率，float
                ):
        self.vocabulary_length = vocabulary_length
        self.embedding_dimension = embedding_dimension
        self.wordvectors = wordvectors
        self.filternumber = filternumber
        self.filterlength = filterlength
        self.dropoutp = dropoutp
        self.classesnumber = classesnumber
        self.learningrate = learningrate
