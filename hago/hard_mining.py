#!/usr/bin/env python
#coding:utf-8
import sys
import os
import cv2
import json
import random
import numpy as np
from scipy.special import softmax


_keys = [line.split()[0] for line in open('../hago/testset/train_0610.txt.1')]
_labels = {line.split()[0]:int(line.split()[1]) for line in open('../hago/testset/train_0610.txt.1')}


def load_from_npy(pr_name, lb_name):
    data0 = np.load(pr_name)
    data0_l = np.load(lb_name)
    return data0, data0_l


if __name__ == '__main__':
    prename = 'ckpt-hago/sex-0610-moco1-init/ret/train-0610-059000'
    data0, datal = load_from_npy(prename + '-pr.npy', prename + '-lb.npy')
    data0 = softmax(data0, axis=1)
    ss = {}
    for s, i in zip(data0, datal):
        url = _keys[i]
        ss[url] = s[_labels[url]]
    
    data = [[] for i in range(5)]
    for url in _keys:
        if url in ss:
            data[_labels[url]].append([url, ss[url]])
    
    for i in range(5):
        data[i].sort(key=lambda x:x[1])
    
    # cat ../hago/testset/train_0610.txt.1 | awk '{a[$2]+=1}END{for(i in a)print i,a[i]}'
    # 0 24190
    # 1 104809
    # 2 1143332
    # 3 261421
    # 4 20969
    '''ii = 2
    alpha = 0.1 # hard_mining.tmp.txt
    alpha = 0.2 # hard_mining.tmp2.txt
    sidx = int(len(data[ii])*alpha)
    data[ii] = data[ii][:sidx] + [i for i in data[ii][sidx:] if random.random() < alpha]
    ii = 3
    alpha = 0.4
    sidx = int(len(data[ii])*alpha)
    data[ii] = data[ii][:sidx] + [i for i in data[ii][sidx:] if random.random() < alpha]'''
    # TODO, 丢掉分数特别低和特别高的case，增加数据准确率
    # hard_mining.tmp3.txt
    '''ii = 2
    beta = 0.05
    data[ii] = data[ii][int(len(data[ii])*beta):-int(len(data[ii])*beta)]
    alpha = 0.2
    sidx = int(len(data[ii])*alpha)
    data[ii] = data[ii][:sidx] + [i for i in data[ii][sidx:] if random.random() < alpha]
    ii = 3
    beta = 0.025
    data[ii] = data[ii][int(len(data[ii])*beta):-int(len(data[ii])*beta)]
    alpha = 0.4
    sidx = int(len(data[ii])*alpha)
    data[ii] = data[ii][:sidx] + [i for i in data[ii][sidx:] if random.random() < alpha]'''
    # hard_mining.tmp4.txt
    ii = 1
    data[ii] = data[ii] * 5
    
    
    f = open('hard_mining.tmp4.txt', 'w')
    for i in range(5):
        for url, s in data[i]:
            print('%s\t%s' % (url, i), file=f)
    f.close()
