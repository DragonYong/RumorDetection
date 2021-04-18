#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 4/18/21-21:29
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : predict.py
# @Project  : 00PythonProjects


# 导入必要的包
import argparse

import numpy as np
import paddle
import paddle.fluid as fluid

parser = argparse.ArgumentParser()
parser.add_argument('--auther', help='inner batch size', default="TuringEmmy", type=str)
parser.add_argument('--MODEL', help='inner batch size', default=None, type=str)
parser.add_argument('--DICT', help='inner batch size', default=None, type=str)
parser.add_argument('--TEXT1', help='inner batch size', default=None, type=str)
parser.add_argument('--TEXT2', help='inner batch size', default=None, type=str)
parser.add_argument('--TEXT3', help='inner batch size', default=None, type=str)

args = parser.parse_args()

paddle.enable_static()
# # 定义输入数据， lod_level不为0指定输入数据为序列数据
words = fluid.data(name='words', shape=[None, 1], dtype='int64', lod_level=1)
label = fluid.data(name='label', shape=[None, 1], dtype='int64')

# use_cuda为False,表示运算场所为CPU;use_cuda为True,表示运算场所为GPU
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 用训练好的模型进行预测并输出预测结果
# 创建执行器
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
infer_exe.run(fluid.default_startup_program())

# save_path = '/media/turing/D741ADF8271B9526/tmp/aistudio/work/infer_model/'

# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=args.MODEL, executor=infer_exe)


# 获取数据
def get_data(sentence):
    # 读取数据字典
    with open(args.DICT, 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])
    dict_txt = dict(dict_txt)
    # 把字符串数据转换成列表数据
    keys = dict_txt.keys()
    data = []
    for s in sentence:
        # 判断是否存在未知字符
        if not s in keys:
            s = '<unk>'
        data.append(int(dict_txt[s]))
    return data


data = []
# 获取图片数据
data1 = get_data(args.TEXT1)
data2 = get_data(args.TEXT2)
data3 = get_data(args.TEXT3)
data.append(data1)
data.append(data2)
data.append(data3)

# 获取每句话的单词数量
base_shape = [[len(c) for c in data]]

# 生成预测数据
tensor_words = fluid.create_lod_tensor(data, base_shape, place)

# 执行预测
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: tensor_words},
                 fetch_list=target_var)

# 分类名称
names = ['谣言', '非谣言']

# 获取结果概率最大的label
for i in range(len(data)):
    lab = np.argsort(result)[0][i][-1]
    print('预测结果标签为：%d， 分类为：%s， 概率为：%f' % (lab, names[lab], result[0][i][lab]))
