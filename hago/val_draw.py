import argparse
import sys
import torch
import logging
import time
import math
import os
import json
import numpy as np
sys.path.append('/data1/liuyidi/moco')

import torch.nn as nn
from loader import val_cls_loader, uint8_normalize
from tensorboardX import SummaryWriter
from scipy.special import softmax
from val_hago import get_pr_xeqy


Nckpts=8000
Ninterval=1000
DIRNAME='DIRNAME=/data1/liuyidi/model/base'


log_dir = DIRNAME+'/log/val'
ret_file = DIRNAME+'val/val-00'
writer = SummaryWriter(log_dir)
for i in range(0,Nckpts+1000,Nckpts):
    acc,pr_xeqy_avg,pr_xeqy_w_avg = val(list_file,ret_file,log_dir,iter_n)


def val(list_file,ret_file,iter_n):

    predict_score = np.load(ret_file + '-pr.npy')
    predict_score = softmax(predict_score, axis=1)
    groundtruth_label = np.load(ret_file + '-lb.npy')
    
    groundtruth_label = np.array([[int(i) for i in line.strip().split()[1].split('|')]
                                  for line in open(list_file)])[groundtruth_label]
    
    
    predict_label = np.argmax(predict_score, axis=1)
    acc = float(np.sum(predict_label==groundtruth_label)) / len(predict_label)
    ret = []
    for i in range(predict_score.shape[1]):
        #print(i)
        xeqy, m, n, th = get_pr_xeqy(predict_score, groundtruth_label, taget_labels=[i])
        print(i, xeqy, m, n, th) #####################
        ret.append([xeqy, m/n])
    pr_xeqy_w_avg = sum([i[0] * i[1] for i in ret])
    pr_xeqy_avg = sum([i[0] for i in ret]) / len(ret)

    #print([i[0] for i in ret])
    print('acc', acc, 'pr_xeqy_w_avg', pr_xeqy_w_avg, 'pr_xeqy_avg', pr_xeqy_avg)

    if log_dir != 'null':
        writer = SummaryWriter(log_dir)
        writer.add_scalar('val-top1_acc', acc, iter_n)
        if predict_score.shape[1] <= 100:
            writer.add_scalar('val-pr_xeqy_w_avg', pr_xeqy_w_avg, iter_n)
            writer.add_scalar('val-pr_xeqy_avg', pr_xeqy_avg, iter_n)
        writer.close()
    return(acc,pr_xeqy_avg,pr_xeqy_w_avg)

    
