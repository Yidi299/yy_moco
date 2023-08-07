#!/usr/bin/env python3

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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


_FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO, format=_FORMAT, stream=sys.stdout
)

logger = logging.getLogger(__name__)
# logger.info('hvd info, size %s, rank %s, local_rank %s.', hvd.size(), hvd.rank(), hvd.local_rank())


def get_pr_xeqy(predict_score, groundtruth_label, taget_labels, taget_precision='x=y', merge_method='sum', lab_w=None):
    '''
    predict_score: NxP 维的 softmax 输出，
    groundtruth_label: N 维的 groundtruth，
    taget_labels: 要计算 pr 的类别，元素为 [0, P) 范围的 int，可以是单个类别，也可以是多个类别，
    merge_method: 当 taget_labels 是多个类别的时候，从多个类别合并到用于排序的 score 的方法，max 或者 sum，
    lab_w: 因为有时不同类别样本会做不同的降采样，用该参数保证结果和无降采样时相同，
    '''
    if lab_w is None:
        lab_w = np.ones(np.max(groundtruth_label)+1)
    if len(taget_labels) > 1:
        taget_labels_set = set(taget_labels)
        merge_f = max if merge_method=='max' else (sum if merge_method=='sum' else None)
        ss = [(merge_f([s[i] for i in taget_labels]),
               1 if l in taget_labels_set else 0,
               lab_w[l])
              for s, l in zip(predict_score, groundtruth_label)]
    else:
        ss = [(s[taget_labels[0]],
               1 if l == taget_labels[0] else 0,
               lab_w[l])
              for s, l in zip(predict_score, groundtruth_label)]
    ss.sort(key=lambda s:-s[0])
    
    pr = []
    m = sum([w for s, l, w in ss if l == 1])
    n = 0
    t = 0
    for s, l, w in ss:
        if l == 1:
            t += w
        n += w
        pr.append([t/m if m>0 else 0, t/n if n>0 else 0, s])
    
    xeqy = []
    th = []
    for x, y, s in pr:
        if taget_precision == 'x=y':
            if min(x, y) > 0 and max(x, y) / min(x, y) < 1.05:
                xeqy.append(x + y)
                th.append(s)
        else:
            if y > 0 and max(y, taget_precision) / min(y, taget_precision) < 1.05:
                xeqy.append(x + x)
                th.append(s)
    xeqy = sum(xeqy) / len(xeqy) / 2 if len(xeqy) > 0 else 0
    th = sum(th) / len(th) if len(th) > 0 else 1.0
    
    return xeqy, m, n, th


def main():
    list_file = sys.argv[1]
    ret_file = sys.argv[2]
    # list_file = '/data1/liuyidi/scene_cls/distill_2/val_list/guaping_2.txt'
    # ret_file = '/data1/liuyidi/scene_cls/distill_2/distill_log_dir/distill_fix/val_guaping/val-016000'
    try:
        log_dir = sys.argv[3]
        iter_n = int(sys.argv[4])
    except:
        log_dir = 'null'
        # log_dir = '/data1/liuyidi/scene_cls/distill_2/distill_log_dir/distill_fix/log/val_guaping/'
        iter_n = 0
    try:
        num_classes = json.loads(sys.argv[5])
    except:
        num_classes = 43
    try:
        num_start = int(sys.argv[6])
    except:
        num_start = 0
    
    ret_file = ret_file.split(',')
    ret = ret_file[0]
    predict_score_all = None
    for ret in ret_file:
        predict_score = np.load(ret + '-pr.npy')
        if num_classes is None:
            predict_score = softmax(predict_score, axis=1)
        elif isinstance(num_classes, int):
            predict_score = softmax(predict_score[:, num_start:num_start+num_classes], axis=1)
        elif isinstance(num_classes, list) and all([isinstance(i, int) for i in num_classes]):
            assert predict_score.shape[1] == sum(num_classes)
            j = 0
            for n in num_classes:
                predict_score[:,j:j+n] = softmax(predict_score[:, j:j+n], axis=1)
                j = j+n
        groundtruth_label = np.load(ret + '-lb.npy')
        if predict_score_all is None:
            n = np.max(groundtruth_label)
            predict_score_all = np.zeros((n + 1, predict_score.shape[1]))
            predict_score_all_n = np.zeros(n + 1)
        for l, p in zip(groundtruth_label, predict_score):
            if l < predict_score_all.shape[0]:
                predict_score_all[l] += p
                predict_score_all_n[l] += 1
    idx = (predict_score_all_n != 0)
    predict_score = predict_score_all[idx] / np.expand_dims(predict_score_all_n[idx], axis=1)
    groundtruth_label = np.arange(predict_score_all.shape[0])[idx]
    
    groundtruth_label = np.array([[int(i) for i in line.strip().split()[1].split('|')]
                                  for line in open(list_file)])[groundtruth_label]
    if len(groundtruth_label[0]) == 1:
        groundtruth_label = [i[0] for i in groundtruth_label]
    else:
        assert isinstance(num_classes, list) and len(num_classes) == len(groundtruth_label[0])
    
    if isinstance(num_classes, list):
        groundtruth_label_raw = groundtruth_label
        predict_score_raw = predict_score
        num_classes_j = 0
        if log_dir != 'null':
            writer = SummaryWriter(log_dir)
        for num_classes_i in range(len(num_classes)):
            n = num_classes[num_classes_i]
            groundtruth_label = groundtruth_label_raw[:, num_classes_i]
            predict_score = predict_score_raw[:, num_classes_j:num_classes_j+n]
            num_classes_j = num_classes_j+n

            predict_label = np.argmax(predict_score, axis=1)
            
            # for ind in range(len(groundtruth_label)):
            #     if groundtruth_label[ind] == 17 or groundtruth_label[ind] ==18 :
            #         groundtruth_label[ind] = 17
            #     elif groundtruth_label[ind] == 19:
            #         groundtruth_label[ind] = 18
            #     elif groundtruth_label[ind] == 20 or groundtruth_label[ind] == 21:
            #         groundtruth_label[ind] = 19
            #     elif groundtruth_label[ind] > 21:
            #         groundtruth_label[ind] -= 2
            
            acc = float(np.sum(predict_label==groundtruth_label)) / len(predict_label)

            if predict_score.shape[1] <= 100:
                ret = []
                # for i in range(predict_score.shape[1]):
                #     #print(i)
                #     xeqy, m, n, th = get_pr_xeqy(predict_score, groundtruth_label, taget_labels=[i])
                #     print(i, xeqy, m, n, th) #####################
                #     ret.append([xeqy, m/n])
                # pr_xeqy_w_avg = sum([i[0] * i[1] for i in ret])
                # pr_xeqy_avg = sum([i[0] for i in ret]) / len(ret)

                

                #print([i[0] for i in ret])
                print('acc', acc, 'pr_xeqy_w_avg', pr_xeqy_w_avg, 'pr_xeqy_avg', pr_xeqy_avg)
            else:
                print('acc', acc)

            if log_dir != 'null':
                writer.add_scalar('val-top1_acc-%s' % num_classes_i, acc, iter_n)
                if predict_score.shape[1] <= 100:
                    writer.add_scalar('val-pr_xeqy_w_avg-%s' % num_classes_i, pr_xeqy_w_avg, iter_n)
                    writer.add_scalar('val-pr_xeqy_avg-%s' % num_classes_i, pr_xeqy_avg, iter_n)
        if log_dir != 'null':
            writer.close()
    else:
        predict_label = np.argmax(predict_score, axis=1)


        # 线上模型对齐
        # for ind in range(len(groundtruth_label)):
        #         if groundtruth_label[ind] == 17 or groundtruth_label[ind] ==18 :
        #             groundtruth_label[ind] = 17
        #         elif groundtruth_label[ind] == 19:
        #             groundtruth_label[ind] = 18
        #         elif groundtruth_label[ind] == 20 or groundtruth_label[ind] == 21:
        #             groundtruth_label[ind] = 19
        #         elif groundtruth_label[ind] > 21:
        #             groundtruth_label[ind] -= 2
        


        #########混淆矩阵
        acc = float(np.sum(predict_label==groundtruth_label)) / len(predict_label)
        
        # pl = predict_label.tolist()
        # C = confusion_matrix(groundtruth_label, pl)
        # plt.matshow(C, cmap=plt.cm.Reds) # 根据最下面的图按自己需求更改颜色
        # for i in range(len(C)):
        #     for j in range(len(C)):
        #         plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        # plt.savefig('/data1/liuyidi/scene_cls/distill_2/fig/'+log_dir.split('/')[-2])        

        if predict_score.shape[1] <= 100:
            ret = []
            for i in range(predict_score.shape[1]):
                #print(i)
                xeqy, m, n, th = get_pr_xeqy(predict_score, groundtruth_label, taget_labels=[i])
                print(i, xeqy, m, n, th) #####################
                ret.append([xeqy, m/n])
            pr_xeqy_w_avg = sum([i[0] * i[1] for i in ret])
            pr_xeqy_avg = sum([i[0] for i in ret]) / len(ret)

            #######一级指标
            # sp = list(range(43))
            # sp1 = [sp[0:14],sp[14:19],sp[19:22],sp[22:29],sp[29:36],sp[36:39],[sp[39]],[sp[40]],sp[41:43]]
            # # sp = list(range(41))
            # # sp1 = [sp[0:14],sp[14:18],sp[18:20],sp[20:27],sp[27:34],sp[34:37],[sp[37]],[sp[38]],sp[39:41]]
            # for i in range(9):
            #     #print(i)
            #     xeqy, m, n, th = get_pr_xeqy(predict_score, groundtruth_label, taget_labels=sp1[i])
            #     print(i, xeqy, m, n, th) #####################
            #     ret.append([xeqy, m/n])
            # pr_xeqy_w_avg = sum([i[0] * i[1] for i in ret])
            # pr_xeqy_avg = sum([i[0] for i in ret]) / len(ret)
            #print([i[0] for i in ret])
            print('acc', acc, 'pr_xeqy_w_avg', pr_xeqy_w_avg, 'pr_xeqy_avg', pr_xeqy_avg)

        else:
            print('acc', acc)

        if log_dir != 'null':
            writer = SummaryWriter(log_dir)
            writer.add_scalar('val-top1_acc', acc, iter_n)
            if predict_score.shape[1] <= 100:
                writer.add_scalar('val-pr_xeqy_w_avg', pr_xeqy_w_avg, iter_n)
                writer.add_scalar('val-pr_xeqy_avg', pr_xeqy_avg, iter_n)
            writer.close()
        
        


if __name__ == "__main__":
    main()

