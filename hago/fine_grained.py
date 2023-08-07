#!/usr/bin/env python
#coding:utf-8
import sys
import os
import cv2
import json
import random
import numpy as np
from scipy.special import softmax


if __name__ == '__main__':
    
    _keys = [line.split()[0] for line in open('../hago/testset/train_0617_all.txt.2')]
    _labels = {line.split()[0]:int(line.split()[1]) for line in open('../hago/testset/train_0617_all.txt.1')}
    
    _labels2 = {}
    tags_name = '#1-女胸|#2-女腰|#3-女腿|#4-女背|#5-女臀|#6-女紧身衣|#7-接吻|#8-男上身|#9-男下身|#10-性感其他'.split('|')
    m = 0
    for line in open('data/CaseSet-Data-9.7w.txt'):
        try:
            did, data, tags, t1, t2 = line.strip().split('\t')
        except:
            continue
        data = json.loads(data)
        tags = json.loads(tags)
        url = data['url'][len('http://221.228.110.3:8900/dataset/'):]
        label = [int(tags_name[i] in tags) for i in range(10)]
        _labels2[url] = label
        if sum(label) > 0:
            m += 1
    print(len(_labels2), m)
    
    f = open('data/train_0617_all.fine.txt', 'w')
    for url in _keys:
        print('%s\t%s\t%s' % (url, _labels[url], ','.join([str(i) for i in _labels2[url]])
                              if url in _labels2 else ','.join(['-1']*10)), file=f)
    f.close()
