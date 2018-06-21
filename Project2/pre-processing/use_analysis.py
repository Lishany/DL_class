#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 13:44:42 2018

@author: xinyu
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from pandas.core.frame import DataFrame

#train read txt
train=open('/Users/xinyu/documents/DL/project2/Weibo Data/weibo_train_data.txt')
uid = []
mid = []
time = []
forward = []
comment = []
like = []
content = []
for line in train:
    lines = line.strip('\n').split('\t')
    uid.append(lines[0])
    mid.append(lines[1])
    time.append(lines[2])
    forward.append(float(lines[3]))
    comment.append(float(lines[4]))
    like.append(float(lines[5]))
    content.append(lines[6])
tdata={"uid" : uid,
       "mid" : mid,
       "time" : time,
       "forward" : forward,
       "comment" : comment,
       "like" : like,
       "content": content}
tdata=DataFrame(tdata)
tdata['con.len']=tdata['content'].apply(len)

# user 

def max2(x):
    return sorted(x)[max(-2,-len(x))]
grouped=tdata[['forward','comment','like']].groupby(tdata['uid'])
user=grouped.mean()
user['count']=grouped.count().iloc[:,1].values
user[['f.max','c.max','l.max']]=grouped.max()
user[['f.2max','c.2max','l.2max']]=grouped.agg(max2)
user[['f.min','c.min','l.min']]=grouped.min()
user[['f.var','c.var','l.var']]=grouped.var()
groupc=tdata['con.len'].groupby(tdata['uid'])
user['con.len.avg']=groupc.mean()
user=user.sort_index(by='count',ascending=False)
user.head()
user.to_csv('/Users/xinyu/documents/DL/project2/Weibo Data/user_ana.csv')
