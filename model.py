#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 2021/4/17-23:40
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : model.py
# @Project  : 00PythonProjects
# 定义长短期记忆网络
from paddle import fluid


def lstm_net(ipt, input_dim):
    # 以数据的IDs作为输入

    emb = fluid.layers.embedding(input=ipt, size=[input_dim, 128], is_sparse=True)

    # 第一个全连接层

    fc1 = fluid.layers.fc(input=emb, size=128)

    # 进行一个长短期记忆操作

    lstm1, _ = fluid.layers.dynamic_lstm(input=fc1,  # 返回：隐藏状态（hidden state），LSTM的神经元状态

                                         size=128)  # size=4*hidden_size

    # 第一个最大序列池操作

    fc2 = fluid.layers.sequence_pool(input=fc1, pool_type='max')

    # 第二个最大序列池操作

    lstm2 = fluid.layers.sequence_pool(input=lstm1, pool_type='max')

    # 以softmax作为全连接的输出层，大小为2,也就是正负面

    out = fluid.layers.fc(input=[fc2, lstm2], size=2, act='softmax')

    return out


