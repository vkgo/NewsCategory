import argparse
import os
import train as T
import test

# 获取args
parser = argparse.ArgumentParser(description='新闻分类')
parser.add_argument('-t', '--task',
                    help='输入要进行的任务:\ntrain 训练\ttest 测试\twebtest GUI web网页测试',
                    type=str,
                    default='train'
                    )
parser.add_argument('-g', '--gpu',
                    help='是否使用GPU运行，\n0: 使用CPU运行\n1: 使用GPU运行\n(webtest GPU运行，若要使用CPU运行，请在webtest.py内修改)',
                    type=int,
                    default=1,
                    choices=[0, 1]
                    )
parser.add_argument('-e', '--epoch',
                    help='训练轮数，测试任务忽略',
                    type=int,
                    default=20
                    )
args = parser.parse_args()



if __name__ == '__main__':

    if args.task == 'train':
        T.train(args)
    elif args.task == 'test':
        test.test(args.gpu)
    elif args.task == 'webtest':
        os.system("streamlit run webtest.py")