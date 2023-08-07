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


def get_data(list_txt_file, ret_npy_file, pr_range=None, pr_map_f=None, merge_mp4=False, merge_m='max'):
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
        if merge_m == 'vag':
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
        elif merge_m == 'max':
            data2 = []
            mp4_i = {}
            for n, l, p in data:
                name = n.split('.mp4')[0] + '.mp4'
                if name not in mp4_i:
                    mp4_i[name] = len(data2)
                    data2.append([name, l, []])
                item = data2[mp4_i[name]]
                assert item[1] == l
                item[2] += [p]
            for item in data2:
                maxs = -1
                maxi = -1
                for i, p in enumerate(item[2]):
                    if p is not None and p[item[1]] > maxs:
                        maxs = p[item[1]]
                        maxi = i
                if maxi >= 0:
                    item[2] = item[2][maxi]
                else:
                    item[2] = None
            data = data2
    return data, PDIM


def main():
    list_file = sys.argv[1]
    ret_file = sys.argv[2]
    try:
        log_dir = sys.argv[3]
        iter_n = int(sys.argv[4])
    except:
        log_dir = 'null'
        iter_n = 0
    try:
        num_classes = int(sys.argv[5])
    except:
        num_classes = 0
    
    data, PDIM = get_data(list_file, ret_file, merge_mp4=1)
    
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
    
    if log_dir != 'null':
        writer = SummaryWriter(log_dir)
        writer.add_scalar('val-top1_acc', acc, iter_n)
        if predict_score.shape[1] <= 100:
            writer.add_scalar('val-pr_xeqy_w_avg', pr_xeqy_w_avg, iter_n)
            writer.add_scalar('val-pr_xeqy_avg', pr_xeqy_avg, iter_n)
        writer.close()


if __name__ == "__main__":
    main()

'''
python hago/val_hago-scene_cls.py ../yylive/scene_cls/val6b1.txt ckpt-yylive/scene_cls-5b-2/val/val-6b1-064000
python hago/val_hago-scene_cls.py ../yylive/scene_cls/val6b1.txt ckpt-yylive/scene_cls-6b/val/val-064000
python hago/val_hago-scene_cls.py ../yylive/scene_cls/val6b1.txt ckpt-yylive/multi_label-1c-fixres/val/val-val6b1-008000
python hago/val_hago-scene_cls.py ../yylive/scene_cls/val6b1.txt ckpt-yylive/scene-game-2/val/val-6b1-102000
'''