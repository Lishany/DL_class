import sys
from pandas import DataFrame
import pandas as pd
import json
dict_data = {}
with open('/Users/lishanyu/Desktop/HW2/weibo_predict_data.txt', 'r')as df:
    # 读每一行
    for line in df:
        # 如果这行是换行符就跳过，这里用'\n'的长度来找空行
        if line.count('\n') == len(line):
            continue
            # 对每行清除前后空格（如果有的话），然后用"："分割
        kv = line.split('\t')
        #print(kv)
        dict_data.setdefault("user", []).append(kv[0])
        dict_data.setdefault("weibo", []).append(kv[1])
        dict_data.setdefault("time", []).append(kv[2])
        #dict_data.setdefault("text", []).append(kv[3])
with open('/Users/lishanyu/Desktop/HW2/predict_jieba.txt', 'r')as df:
    for line in df:
        dict_data.setdefault("text", []).append(line)

columnsname=list(dict_data.keys())

dict_tdata = {}
with open('/Users/lishanyu/Desktop/HW2/weibo_train_data.txt', 'r')as df:
    # 读每一行
    for line in df:
        # 如果这行是换行符就跳过，这里用'\n'的长度来找空行
        if line.count('\n') == len(line):
            continue
            # 对每行清除前后空格（如果有的话），然后用"："分割
        kv = line.split('\t')
        #print(kv)
        dict_tdata.setdefault("user", []).append(kv[0])
        dict_tdata.setdefault("weibo", []).append(kv[1])
        dict_tdata.setdefault("time", []).append(kv[2])
        dict_tdata.setdefault("label1", []).append(kv[3])
        dict_tdata.setdefault("label2", []).append(kv[4])
        dict_tdata.setdefault("label3", []).append(kv[5])
        #dict_tdata.setdefault("text", []).append(kv[6])
with open('/Users/lishanyu/Desktop/HW2/train_jieba.txt', 'r')as df:
    for line in df:
        dict_tdata.setdefault("text", []).append(line)
columnsname=list(dict_tdata.keys())
len(dict_tdata['user'])
len(dict_tdata['weibo'])


dict_tdata["text"][1229618+11364]

#frame = DataFrame(dict_data, columns=columnsname)
with open("/Users/lishanyu/Desktop/HW2/test_jeiba.json",'w',encoding='utf-8') as json_file:
    json.dump(dict_data,json_file,ensure_ascii=False)

with open("/Users/lishanyu/Desktop/HW2/train_jeiba.json",'w',encoding='utf-8') as json_file:
    json.dump(dict_tdata,json_file,ensure_ascii=False)
