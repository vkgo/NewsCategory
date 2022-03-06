import torch
from torch.utils.tensorboard import SummaryWriter

import Module as M
import loaddataset
from torchtext.legacy.data import BucketIterator
import time
from torch import nn


def train(args):
        # 设置batch_sizes
        batch_sizes = 128
        # 判断cuda是否可用
        if args.gpu == 1 and torch.cuda.is_available() == False:
                print("cuda无法调用，以下使用CPU运行")
                args.gpu = 0

        # 获取新闻数据，总数据，还没有形成数据集
        print("开始载入数据")
        starttime = time.time()
        traindata, valdata, testdata, config = loaddataset.load()
        print("载入成功，用时:{}".format(time.time() - starttime))
        testdatalength = len(testdata)

        # 生成数据集
        print("开始生成数据集")
        starttime = time.time()
        traindataset, valdataset, testdataset = BucketIterator.splits(
                (traindata, valdata, testdata),  # 指定生成迭代器的数据集
                batch_sizes=(batch_sizes, batch_sizes, batch_sizes),  # batch size
                sort_key=lambda x: len(x.Text)  # 按句子的长度来排列batch，把句子长度相近的放在同一个batch里面
        )
        print("生成成功，用时:{}".format(time.time() - starttime))

        # 创建模型对象
        TextCNN = M.TextCNN(config)
        if args.gpu == 1:
                TextCNN = TextCNN.cuda()

        # 损失函数
        loss_fn = nn.CrossEntropyLoss()
        if args.gpu == 1:
                loss_fn = loss_fn.cuda()

        # 使用SGD优化器
        optim = torch.optim.SGD(params=TextCNN.parameters(), lr=config.learningrate)

        # 使用TensorBoard查看数据
        writer = SummaryWriter("../logs")


        print("----------开始训练----------")
        for epoch in range(args.epoch):
                print("***第{}轮训练***".format(epoch + 1))
                start_time = time.time()
                counter = 0
                round_loss = 0.0
                TextCNN.train()
                for batches in traindataset:
                        texts = batches.Text
                        labels = batches.Label
                        if args.gpu == 1:
                                texts = texts.cuda()
                                labels = labels.cuda()

                        counter += 1

                        module_output = TextCNN(texts)
                        loss_result = loss_fn(module_output, labels)

                        optim.zero_grad() # 梯度清零
                        loss_result.backward() # 反向传播
                        optim.step() # 生效

                        round_loss += loss_result.item()

                        if counter % 100 == 0:
                                end_time = time.time()
                                print(end_time - start_time)
                                print("第{}次训练 平均损失{}".format(counter, round_loss / 100))
                                writer.add_scalar(tag="Train_Loss", scalar_value=round_loss / 100,
                                                  global_step=counter)
                                round_loss = 0.0
                                start_time = time.time()

                # 结束一个epoch，检验正确率
                TextCNN.eval()
                with torch.no_grad():
                        total_loss = 0.0
                        counter = 0
                        correct_num = 0 # 总共正确的个数

                        for batches in testdataset:
                                texts = batches.Text
                                labels = batches.Label
                                if args.gpu == 1:
                                        texts = texts.cuda()
                                        labels = labels.cuda()
                                counter += 1
                                module_output = TextCNN(texts)
                                loss_result = loss_fn(module_output, labels)
                                total_loss += loss_result.item()

                                correct_num += (module_output.argmax(1) == labels).sum().item()

                        print("##验证集每batches平均损失:{}".format(total_loss / counter))
                        print("##验证集每batches平均正确率:{}".format(correct_num / testdatalength))
                        writer.add_scalar(tag="正确率", scalar_value=correct_num / testdatalength, global_step=epoch + 1)

                # 保存模型
                if epoch % 8 == 0:
                        SavedLoc = "../SavedModel/mymodule_{}.pth".format(epoch)
                        # SavedLoc = "../SavedModel/temp.pth"
                        # TextCNN = TextCNN.cpu()
                        torch.save(TextCNN, SavedLoc)

