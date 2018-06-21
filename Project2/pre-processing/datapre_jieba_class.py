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
        dict_tdata.setdefault("text", []).append(kv[6])
with open('/Users/lishanyu/Desktop/HW2/train_jieba.txt', 'r')as df:
    for line in df:
        dict_tdata.setdefault("text", []).append(line)
columnsname=list(dict_tdata.keys())
len(dict_tdata['user'])
len(dict_tdata['weibo'])
#frame = DataFrame(dict_data, columns=columnsname)
with open("/Users/lishanyu/Desktop/HW2/test_jeiba.json",'w',encoding='utf-8') as json_file:
    json.dump(dict_data,json_file,ensure_ascii=False)

with open("/Users/lishanyu/Desktop/HW2/train_jeiba.json",'w',encoding='utf-8') as json_file:
    json.dump(dict_tdata,json_file,ensure_ascii=False)

textt = []
with open('/Users/lishanyu/Desktop/HW2/train_jieba.txt', 'r')as df:
    for line in df:
        textt.append(line)

flow = []
with open('/Users/lishanyu/Desktop/HW2/classlabel.txt','r') as df:
    for line in df:
        kv = line.split('\t')
        flow.append(int(kv[1]))
dict_tdata1 = {}
dict_tdata2 = {}
dict_tdata3 = {}
dict_tdata4 = {}
dict_tdata5 = {}
dict_tdata6 = {}
dict_tdata7 = {}
dict_tdata = {}

with open('/Users/lishanyu/Desktop/HW2/weibo_train_data.txt', 'r')as df:
    # 读每一行
    count = 0
    for line in df:
        # 如果这行是换行符就跳过，这里用'\n'的长度来找空行
        if line.count('\n') == len(line):
            continue
            # 对每行清除前后空格（如果有的话），然后用"："分割
        kv = line.split('\t')
        #print(kv)
        if flow[count] == 0:
            count = count + 1
            continue
        if flow[count] == 1:
            dict_tdata = dict_tdata1
        elif flow[count] == 2:
            dict_tdata = dict_tdata2
        elif flow[count] == 3:
            dict_tdata = dict_tdata3
        elif flow[count] == 4:
            dict_tdata = dict_tdata4
        elif flow[count] == 5:
            dict_tdata = dict_tdata5
        elif flow[count] == 6:
            dict_tdata = dict_tdata6
        elif flow[count] == 7:
            dict_tdata = dict_tdata7
        dict_tdata.setdefault("user", []).append(kv[0])
        dict_tdata.setdefault("weibo", []).append(kv[1])
        dict_tdata.setdefault("time", []).append(kv[2])
        dict_tdata.setdefault("label1", []).append(kv[3])
        dict_tdata.setdefault("label2", []).append(kv[4])
        dict_tdata.setdefault("label3", []).append(kv[5])
        dict_tdata.setdefault("text", []).append(textt[count])
        count = count+1

len(textt)
count
len(dict_tdata2["user"])
len(dict_tdata2["text"])
len(dict_tdata7["user"])
dict_tdata2["text"][0]
dict_tdata2["user"][0]
with open("/Users/lishanyu/Desktop/HW2/train_jeiba7/train_jeiba1.json",'w',encoding='utf-8') as json_file:
    json.dump(dict_tdata1,json_file,ensure_ascii=False)
with open("/Users/lishanyu/Desktop/HW2/train_jeiba7/train_jeiba2.json",'w',encoding='utf-8') as json_file:
    json.dump(dict_tdata2,json_file,ensure_ascii=False)
with open("/Users/lishanyu/Desktop/HW2/train_jeiba7/train_jeiba3.json",'w',encoding='utf-8') as json_file:
    json.dump(dict_tdata3,json_file,ensure_ascii=False)
with open("/Users/lishanyu/Desktop/HW2/train_jeiba7/train_jeiba4.json",'w',encoding='utf-8') as json_file:
    json.dump(dict_tdata4,json_file,ensure_ascii=False)
with open("/Users/lishanyu/Desktop/HW2/train_jeiba7/train_jeiba5.json",'w',encoding='utf-8') as json_file:
    json.dump(dict_tdata5,json_file,ensure_ascii=False)
with open("/Users/lishanyu/Desktop/HW2/train_jeiba7/train_jeiba6.json",'w',encoding='utf-8') as json_file:
    json.dump(dict_tdata6,json_file,ensure_ascii=False)
with open("/Users/lishanyu/Desktop/HW2/train_jeiba7/train_jeiba7.json",'w',encoding='utf-8') as json_file:
    json.dump(dict_tdata7,json_file,ensure_ascii=False)
