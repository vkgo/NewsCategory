# NewsCategory
Chinese news classification based on TextCNN 基于TextCNN的中文新闻分类 数据结构大作业
#


# 操作演示

# 4.1	 命令及其含义
输入python run.py <指令>可以进行程序的功能调用，其中—help可以调出帮助界面(Figure 1)。

 ![image](https://user-images.githubusercontent.com/41435573/166198221-f9fa2337-5bb6-4900-a5dd-f4e70fb59d1c.png)
 
-t或—task，任务，代表所要程序进行的任务，可以输入的范围在{train, test, webtest}之中。train表示进行模型训练，test表示进行测试，webtest表示使用带web GUI界面的测试工具。
-g或—gpu，使用GPU，代表程序在训练train或者测试test任务中是否使用GPU，输入范围在{0, 1}之中，1代表使用GPU，0代表不使用。
-e或—epoch，epoch数，代表训练轮数。

    使用示范:
    python run.py -t train -g 1 -e 32
    #进行训练，使用GPU，epoch为32
    python run.py -t test -g 1
    #进行测试，使用GPU
    python run.py -t webtest
    #进行测试，使用web GUI界面（默认使用GPU）


# 4.2	 文件用途解释
![image](https://user-images.githubusercontent.com/41435573/166198335-0596fbd9-fca9-4011-9bf4-cee2c2118f0f.png)

data文件夹用于存放数据集，预训练词向量，停用词。
logs文件夹用于存储tensorboard的日志文件，用于程序开发的时候实现参数一定程度上的可视化。
obj文件夹用于存储加载完数据之后所生成的字典文件，用于将训练时生成的词表字典、标签字典在测试时使用。
SavedModel文件夹用于存储训练时生成的模型，每epoch保存一次。
src文件夹存放源码数据。	


![image](https://user-images.githubusercontent.com/41435573/166198599-622c4aa1-f876-48ee-bd12-081ffd8061ac.png)

其中：
config.py——配置文件，存储项目大部分的配置参数
forecast.py——预测类，集成了分类预测功能
loaddataset——加载数据，将数据集的数据从硬盘中加载到程序
Module.py——模型
objprocess——对象操作，保存对象到硬盘或从硬盘读取对象
run.py——导航页，项目主程序
test.py——测试
webtest——web GUI界面测试
train.py——训练



# 4.3 训练模型
# 4.3.1 使用默认配置训练
进入src目录，python run.py命令直接运行，若要设定运行模式，可见4.1命令及其含义 。

![image](https://user-images.githubusercontent.com/41435573/166198681-d741f134-30b8-4d7e-ace5-3889ce01f9d7.png)

# 4.3.2 自定义训练
1.	停用词
设置位置：src/loaddataset.py
于jiebatokensize函数可以设置加载的停用词文件
 
![image](https://user-images.githubusercontent.com/41435573/166198730-030d7dcc-023f-4e89-9f79-24b46f405cb0.png)

2.	数据集
设置位置：src/loaddataset.py
于load函数中，可以设置加载的数据集（训练集、验证集、测试集）
 
![image](https://user-images.githubusercontent.com/41435573/166198776-edeef919-8975-4b67-94de-939ed6b41681.png)

训练集的格式须为tsv，且是<标签> <新闻内容>顺序。
 
![image](https://user-images.githubusercontent.com/41435573/166198796-822b1428-d060-41db-a819-199b93ea359e.png)

3.	模型设置
设置位置：src/config.py
类初始化函数中带默认值的是可调参数，不带默认值的是利用训练过程中生成的数据进行赋值（Figure 8）。
模型中人为可调的参数及其含义：
	dropoutp——dropout层在训练中神经元失活的概率
	filterlength——卷积层卷积核大小，[3 4 5]表示有3种不同大小的卷积核，大小分别是3*embedding_dimension、4*embedding_dimension、5*embedding_dimension
	filternumber——同一种卷积核的数量
	learningrate——学习速率
 
![image](https://user-images.githubusercontent.com/41435573/166198818-508e21dc-12ce-44a5-8b40-269dafdba70f.png)

# 4.4 测试
1. 使用命令行测试
进入src目录，python run.py -t test命令直接运行，若要设定运行模式，可见4.1命令及其含义 。在输入框输入所要进行测试的新闻内容便可以得到分类结果。
 
![image](https://user-images.githubusercontent.com/41435573/166198842-d2acc731-d247-4698-bb4f-8b5f63c59ed5.png)

2. 使用web GUI界面测试
进入src目录，python run.py -t webtest命令直接运行。
 
![image](https://user-images.githubusercontent.com/41435573/166198869-4aa15ea9-c3e9-48da-a4f1-ba341c94e4c1.png)

进入提示的局域网地址后，在文本框里面输入要进行预测的新闻，点击确认即可得到预测结果。
 
![image](https://user-images.githubusercontent.com/41435573/166198888-41a9534f-a5b5-4d6d-96ee-011b87bf949a.png)
