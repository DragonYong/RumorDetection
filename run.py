#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 2021/4/17-23:38
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : run.py.py
# @Project  : 00PythonProjects

# 导入必要的包
import argparse
import os
from multiprocessing import cpu_count

import paddle
# 生成数据字典
from paddle import fluid

from model import lstm_net

parser = argparse.ArgumentParser()
parser.add_argument('--auther', help='auther name', default="TuringEmmy", type=str)
parser.add_argument('--DATA_LIST', help='input raw data', default=None, type=str)
parser.add_argument('--MODEL', help='model save to path', default=None, type=str)
parser.add_argument('--BATCH_SIZE', help='inner batch size', default=1, type=int)
parser.add_argument('--EPOCH_NUM', help='how manny epoch you need', default=1, type=int)
parser.add_argument('--IS_DRAW', help='weather to show graph', action='store_true', default=False)

args = parser.parse_args()


def create_dict(data_path, dict_path):
    dict_set = set()
    # 读取全部数据
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 把数据生成一个元组
    for line in lines:
        content = line.split('\t')[-1].replace('\n', '')
        for s in content:
            dict_set.add(s)
    # 把元组转换成字典，一个字对应一个数字
    dict_list = []
    i = 0
    for s in dict_set:
        dict_list.append([s, i])
        i += 1
    # 添加未知字符
    dict_txt = dict(dict_list)
    end_dict = {"<unk>": i}
    dict_txt.update(end_dict)
    # 把这些字典保存到本地中
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(str(dict_txt))
    print("数据字典生成完成！")


# 获取字典的长度
def get_dict_len(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        line = eval(f.readlines()[0])
    return len(line.keys())


# 创建序列化表示的数据,并按照一定比例划分训练数据与验证数据
def create_data_list(data_list_path):
    # 在生成数据之前，首先将eval_list.txt和train_list.txt清空
    with open(os.path.join(data_list_path, 'eval_list.txt'), 'w', encoding='utf-8') as f_eval:
        f_eval.seek(0)
        f_eval.truncate()

    with open(os.path.join(data_list_path, 'train_list.txt'), 'w', encoding='utf-8') as f_train:
        f_train.seek(0)
        f_train.truncate()

    with open(os.path.join(data_list_path, 'dict.txt'), 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])

    with open(os.path.join(data_list_path, 'all_data.txt'), 'r', encoding='utf-8') as f_data:
        lines = f_data.readlines()

    i = 0
    with open(os.path.join(data_list_path, 'eval_list.txt'), 'a', encoding='utf-8') as f_eval, open(
            os.path.join(data_list_path, 'train_list.txt'), 'a', encoding='utf-8') as f_train:
        for line in lines:
            words = line.split('\t')[-1].replace('\n', '')
            label = line.split('\t')[0]
            labs = ""
            if i % 8 == 0:
                for s in words:
                    lab = str(dict_txt[s])
                    labs = labs + lab + ','
                labs = labs[:-1]
                labs = labs + '\t' + label + '\n'
                f_eval.write(labs)
            else:
                for s in words:
                    lab = str(dict_txt[s])
                    labs = labs + lab + ','
                labs = labs[:-1]
                labs = labs + '\t' + label + '\n'
                f_train.write(labs)
            i += 1

    print("数据列表生成完成！")


# data_list_path = "/media/turing/D741ADF8271B9526/tmp/aistudio/data/"
all_data_path = args.DATA_LIST + "all_data.txt"
# dict_path为数据字典存放路径
dict_path = args.DATA_LIST + "dict.txt"

# 创建数据字典，存放位置：dict.txt。在生成之前先清空dict.txt
with open(dict_path, 'w') as f:
    f.seek(0)
    f.truncate()
create_dict(all_data_path, dict_path)

# 创建数据列表，存放位置：train_list.txt eval_list.txt
create_data_list(args.DATA_LIST)


def data_mapper(sample):
    data, label = sample
    data = [int(data) for data in data.split(',')]
    return data, int(label)


# 定义数据读取器
def data_reader(data_path):
    def reader():
        with open(data_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data, label = line.split('\t')
                yield data, label

    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 1024)


# 获取训练数据读取器和测试数据读取器
# BATCH_SIZE = 128

train_list_path = args.DATA_LIST + 'train_list.txt'
eval_list_path = args.DATA_LIST + 'eval_list.txt'

train_reader = paddle.batch(
    reader=data_reader(train_list_path),
    batch_size=args.BATCH_SIZE)
eval_reader = paddle.batch(
    reader=data_reader(eval_list_path),
    batch_size=args.BATCH_SIZE)

paddle.enable_static()

# 定义输入数据， lod_level不为0指定输入数据为序列数据
words = fluid.data(name='words', shape=[None, 1], dtype='int64', lod_level=1)
label = fluid.data(name='label', shape=[None, 1], dtype='int64')

# 获取数据字典长度
dict_dim = get_dict_len(dict_path)

model = lstm_net(words, dict_dim)

# 获取损失函数和准确率
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 获取预测程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.001)
opt = optimizer.minimize(avg_cost)

# use_cuda为False,表示运算场所为CPU;use_cuda为True,表示运算场所为GPU
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义数据映射器
feeder = fluid.DataFeeder(place=place, feed_list=[words, label])

all_train_iter = 0
all_train_iters = []
all_train_costs = []
all_train_accs = []

all_eval_iter = 0
all_eval_iters = []
all_eval_costs = []
all_eval_accs = []

from matplotlib import pyplot as plt


def draw_process(title, iters, costs, accs, label_cost, lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs, color='red', label=label_cost)
    plt.plot(iters, accs, color='green', label=lable_acc)
    plt.legend()
    plt.grid()
    plt.show()


# 开始训练
for pass_id in range(args.EPOCH_NUM):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])
        all_train_iter = all_train_iter + args.BATCH_SIZE
        all_train_iters.append(all_train_iter)
        all_train_costs.append(train_cost[0])
        all_train_accs.append(train_acc[0])

        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Acc:%0.5f' % (pass_id, batch_id, train_cost[0], train_acc[0]))
    # 进行验证
    eval_costs = []
    eval_accs = []
    for batch_id, data in enumerate(eval_reader()):
        eval_cost, eval_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost, acc])
        eval_costs.append(eval_cost[0])
        eval_accs.append(eval_acc[0])

        all_eval_iter = all_eval_iter + args.BATCH_SIZE
        all_eval_iters.append(all_eval_iter)
        all_eval_costs.append(eval_cost[0])
        all_eval_accs.append(eval_acc[0])
        # 计算平均预测损失在和准确率
    eval_cost = (sum(eval_costs) / len(eval_costs))
    eval_acc = (sum(eval_accs) / len(eval_accs))
    print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, eval_cost, eval_acc))

# 保存模型
if not os.path.exists(args.MODEL):
    os.makedirs(args.MODEL)
fluid.io.save_inference_model(args.MODEL,
                              feeded_var_names=[words.name],
                              target_vars=[model],
                              executor=exe)
print('训练模型保存完成！')

if args.IS_DRAW:
    print(args.IS_DRAW)
    draw_process("train", all_train_iters, all_train_costs, all_train_accs, "trainning cost", "trainning acc")
    draw_process("eval", all_eval_iters, all_eval_costs, all_eval_accs, "evaling cost", "evaling acc")
