#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 2021/4/17-23:38
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : data.py
# @Project  : 00PythonProjects
import argparse
import json
import os
import random
# 解压原始数据集，将Rumor_Dataset.zip解压至data目录下
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--auther', help='inner batch size', default="TuringEmmy", type=str)
parser.add_argument('--SRC_PATH', help='inner batch size', default=None, type=str)
parser.add_argument('--TARHET_PATH', help='inner batch size', default=None, type=str)
parser.add_argument('--DATA_LIST', help='inner batch size', default=None, type=str)

args = parser.parse_args()

# src_path = "/media/turing/D741ADF8271B9526/DATA/aistudio/data20519/Rumor_Dataset.zip"
# target_path = "/media/turing/D741ADF8271B9526/tmp/aistudio/data/Chinese_Rumor_Dataset-master"
if (not os.path.isdir(args.TARHET_PATH)):
    z = zipfile.ZipFile(args.SRC_PATH, 'r')
    z.extractall(path=args.TARHET_PATH)
    z.close()

# 分别为谣言数据、非谣言数据、全部数据的文件路径
rumor_class_dirs = os.listdir(args.TARHET_PATH + "/Chinese_Rumor_Dataset-master/CED_Dataset/rumor-repost/")
non_rumor_class_dirs = os.listdir(args.TARHET_PATH + "/Chinese_Rumor_Dataset-master/CED_Dataset/non-rumor-repost/")
original_microblog = args.TARHET_PATH + "/Chinese_Rumor_Dataset-master/CED_Dataset/original-microblog/"

# 谣言标签为0，非谣言标签为1
rumor_label = "0"
non_rumor_label = "1"

# 分别统计谣言数据与非谣言数据的总数
rumor_num = 0
non_rumor_num = 0

all_rumor_list = []
all_non_rumor_list = []

# 解析谣言数据
for rumor_class_dir in rumor_class_dirs:
    if (rumor_class_dir != '.DS_Store'):
        # 遍历谣言数据，并解析
        with open(original_microblog + rumor_class_dir, 'r') as f:
            rumor_content = f.read()
        rumor_dict = json.loads(rumor_content)
        all_rumor_list.append(rumor_label + "\t" + rumor_dict["text"] + "\n")
        rumor_num += 1

# 解析非谣言数据
for non_rumor_class_dir in non_rumor_class_dirs:
    if (non_rumor_class_dir != '.DS_Store'):
        with open(original_microblog + non_rumor_class_dir, 'r') as f2:
            non_rumor_content = f2.read()
        non_rumor_dict = json.loads(non_rumor_content)
        all_non_rumor_list.append(non_rumor_label + "\t" + non_rumor_dict["text"] + "\n")
        non_rumor_num += 1

print("谣言数据总量为：" + str(rumor_num))
print("非谣言数据总量为：" + str(non_rumor_num))

# 全部数据进行乱序后写入all_data.txt

# data_list_path = "/media/turing/D741ADF8271B9526/tmp/aistudio/data/"
all_data_path = args.DATA_LIST + "all_data.txt"

all_data_list = all_rumor_list + all_non_rumor_list

random.shuffle(all_data_list)

# 在生成all_data.txt之前，首先将其清空
with open(all_data_path, 'w') as f:
    f.seek(0)
    f.truncate()

with open(all_data_path, 'a') as f:
    for data in all_data_list:
        f.write(data)
