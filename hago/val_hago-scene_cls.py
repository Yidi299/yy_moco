#!/usr/bin/env python3

import argparse
import sys
import torch
import logging
import time
import math
import os
import numpy as np
sys.path.append('/data/remote/Moco')

import torch.nn as nn
from loader import val_cls_loader, uint8_normalize
from tensorboardX import SummaryWriter
from scipy.special import softmax

from val_hago import get_pr_xeqy

_FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO, format=_FORMAT, stream=sys.stdout
)

logger = logging.getLogger(__name__)
# logger.info('hvd info, size %s, rank %s, local_rank %s.', hvd.size(), hvd.rank(), hvd.local_rank())

if 0:
    idx_map = {
    0: '0-13',
    1: '14-17',
    2: '18-19',
    3: '20-26',
    4: '27-31',
    5: '32-34',
    6: 35,
    7: 36,
    8: 37,}
else:
    idx_map = {
    0: '0-13',
    1: '14-17',
    2: '18-19',
    3: '20-26',
    4: '27-33',
    5: '34-36',
    6: 37,
    7: 38,
    8: '39-40',}
f_map1 = None
f_map2 = None
def merge_cls1_f(l, p=None):
    global f_map1
    global f_map2
    if f_map1 is None:
        f_map1 = {}
        f_map2 = {}
        for i in idx_map:
            j = []
            for k in str(idx_map[i]).split(','):
                if '-' in k:
                    j += list(range(int(k.split('-')[0]), int(k.split('-')[1])+1))
                else:
                    j += [int(k)]
            f_map1[i] = j
            for k in j:
                f_map2[k] = i
    if p is not None:
        assert len(p) == len(f_map2)
        p2 = np.zeros((len(f_map1),), dtype='float32')
        for i in f_map1:
            j = f_map1[i]
            p2[i] = np.sum(p[j], keepdims=False)
        return f_map2[l], p2
    return f_map2[l]


def get_data(list_txt_file, ret_npy_file, pr_range=None, pr_map_f=None, merge_mp4=False, merge_cls1=False):
    # 返回 [[name, label, score]]
    names = [l.split()[:2] for l in open(list_txt_file)]
    labels = [int(i[1]) for i in names]
    names  = [i[0] for i in names]
    data = [[n, l, None] for n, l in zip(names, labels)]
    if isinstance(ret_npy_file, list):
        # 多个模型结果集成
        pr = np.load(ret_npy_file[0] + '-pr.npy')
        if pr_range is not None:
            pr = pr[:, pr_range[0]:pr_range[1]]
        PDIM = pr.shape[1]
        pr_all = np.zeros((len(names), PDIM), dtype='float32')
        pr_n = np.zeros((len(names),), dtype='int64')
        for n, item in enumerate(ret_npy_file):
            n += 1
            pr = np.load(item + '-pr.npy')
            if pr_range is not None:
                pr = pr[:, pr_range[0]:pr_range[1]]
            pr = softmax(pr, axis=1)
            lb = np.load(item + '-lb.npy')
            for l, p in zip(lb, pr):
                if pr_n[l] < n:
                    pr_all[l] += pr
                    pr_n[l] += 1
        pr_n[pr_n==0] = 1
        pr_all *= (1 / pr_n).reshape((len(names), 1))
        if pr_map_f is not None:
            pr_all = [pr_map_f(p) for p in pr_all]
            PDIM = len(pr_all[0])
        for i, p in enumerate(pr_all):
            if sum(p) > 0.9:
                data[i][2] = p
    else:
        pr = np.load(ret_npy_file + '-pr.npy')
        if pr_range is not None:
            pr = pr[:, pr_range[0]:pr_range[1]]
        PDIM = pr.shape[1]
        pr = softmax(pr, axis=1)
        lb = np.load(ret_npy_file + '-lb.npy')
        exp = set()
        if pr_map_f is not None:
            pr = [pr_map_f(p) for p in pr]
            PDIM = len(pr[0])
        for l, p in zip(lb, pr):
            if int(l) not in exp:
                exp.add(int(l))
                data[l][2] = p
    if merge_mp4:
        data2 = []
        mp4_i = {}
        mp4_n = {}
        for n, l, p in data:
            name = n.split('.mp4')[0] + '.mp4'
            if name not in mp4_i:
                mp4_i[name] = len(data2)
                mp4_n[name] = 0
                data2.append([name, l, p])
                if p is not None:
                    mp4_n[name] += 1
            else:
                item = data2[mp4_i[name]]
                assert item[1] == l
                if p is not None:
                    if mp4_n[name] == 0:
                        item[2] = p
                    else:
                        item[2] += p
                    mp4_n[name] += 1
        for item in data2:
            if mp4_n[item[0]] > 0:
                item[2] /= mp4_n[item[0]]
        data = data2
    if merge_cls1:
        data = [[n, *(merge_cls1_f(l, p))] if p is not None else [n, l, p]
                for n, l, p in data]
        for n, l, p in data:
            if p is not None:
                PDIM = len(p)
    return data, PDIM


def p41_to_38(p):
    p2 = np.zeros((38,), dtype='float32')
    p2[:29] = p[:29]
    p2[27] += p[29]
    p2[27] += p[30]
    p2[29:] = p[31:40]
    p2[37] += p[40]
    return p2
def p42_to_41(p):
    p2 = np.zeros((41,), dtype='float32')
    p2[:12] = p[:12]
    p2[12:] = p[13:]
    return p2
def p42_to_38(p):
    return p41_to_38(p42_to_41(p))
def p38_to_41(p):
    p2 = np.zeros((41,), dtype='float32')
    p2[:29] = p[:29]
    p2[31:40] = p[29:]
    return p2
def movie5_to_41(p):
    p2 = np.zeros((41,), dtype='float32')
    p2[40] = p[0]
    p2[27] = p[1]
    p2[28] = p[2]
    p2[31] = p[3]
    p2[32] = p[4]
    return p2
def game8_to_41(p):
    p2 = np.zeros((41,), dtype='float32')
    p2[40] = p[0]
    p2[20:27] = p[1:8]
    return p2


def main():
    list_file = sys.argv[1]
    ret_file = sys.argv[2]
    
    #data, PDIM = get_data(list_file, ret_file, pr_map_f=p38_to_41, merge_mp4=0, merge_cls1=1)
    #data, PDIM = get_data(list_file, ret_file, pr_map_f=None, merge_mp4=0, merge_cls1=0)
    #data, PDIM = get_data(list_file, ret_file, pr_map_f=p42_to_41, merge_mp4=0, merge_cls1=1)
    #data, PDIM = get_data(list_file, ret_file, pr_range=[6, 11], pr_map_f=movie5_to_41, merge_mp4=0, merge_cls1=1)
    data, PDIM = get_data(list_file, ret_file, pr_map_f=game8_to_41, merge_mp4=0, merge_cls1=1)
    
    ret = []
    for i in range(PDIM):
        xeqy, m, n, th = get_pr_xeqy([p for n, l, p in data if p is not None],
                                     [l for n, l, p in data if p is not None],
                                     taget_labels=[i])
        print(i, xeqy, m, n, th)
        ret.append([xeqy, m/n])
    
    acc = len([0 for n, l, p in data if p is not None and np.argmax(p)==l]) / len(
              [0 for n, l, p in data if p is not None])
    pr_xeqy_w_avg = sum([i[0] * i[1] for i in ret])
    pr_xeqy_avg = sum([i[0] for i in ret]) / len(ret)
    
    print('acc', acc, 'pr_xeqy_w_avg', pr_xeqy_w_avg, 'pr_xeqy_avg', pr_xeqy_avg)


if __name__ == "__main__":
    main()

'''
python hago/val_hago-scene_cls.py ../yylive/scene_cls/val6b1.txt ckpt-yylive/scene_cls-5b-2/val/val-6b1-064000
python hago/val_hago-scene_cls.py ../yylive/scene_cls/val6b1.txt ckpt-yylive/scene_cls-6b/val/val-064000
python hago/val_hago-scene_cls.py ../yylive/scene_cls/val6b1.txt ckpt-yylive/multi_label-1c-fixres/val/val-val6b1-008000
python hago/val_hago-scene_cls.py ../yylive/scene_cls/val6b1.txt ckpt-yylive/scene-game-2/val/val-6b1-102000
'''