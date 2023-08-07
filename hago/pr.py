#!/usr/bin/env python
#coding:utf-8
import sys
import os
import cv2
import json
import random
import numpy as np
from scipy.special import softmax


# _keys = [line.split()[0] for line in open('../hago/testset/test_6_4.txt.1')]
# _labels = {line.split()[0]:int(line.split()[1]) for line in open('../hago/testset/test_6_4.txt.1')}
_keys = [line.split()[0] for line in open('../hago/daily-tag/diaoxing-img.txt')]
_labels = {line.split()[0]:int(line.split()[1]) for line in open('../hago/daily-tag/diaoxing-img.txt')}


def load_from_npy(pr_name, lb_name):
    print(pr_name)
    data0 = np.load(pr_name)
    data0_l = np.load(lb_name)
    print(data0.shape, data0_l.shape, file=sys.stderr)
    datal = [_labels[_keys[i]] for i in data0_l]
    return data0, datal


def get_pr(data0, datal, off, cc=0, ccl=[0], ww=[1, 10, 1]):
    ss = softmax(data0 + np.array(off).reshape((1, len(off))), axis=1)[:, cc]
    ss = [(ss[i], datal[i]) for i in range(len(ss))]
    ss.sort(key=lambda s:-s[0])
    m = sum([ww[s[1]] for s in ss if s[1] in ccl])
    print(m)
    n = 0
    t = 0
    pr = []
    for j, s in enumerate(ss):
        n0 = ww[s[1]]
        if s[1] in ccl:
            t += n0
        n += n0
        pr.append([t/m, t/n])
    return pr


def pr_to_log(writer, prename):
    data0, datal = load_from_npy(prename + '-pr.npy', prename + '-lb.npy')
    # data0 = data0[:, :5]
    off = [0, 0, 0]
    ret = get_pr(data0, datal, off)
    xeqy = []
    for x, y in ret:
        if min(x, y) == 0:
            xeqy.append(0)
        elif max(x, y) / min(x, y) < 1.05:
            xeqy.append(x + y)
    xeqy = sum(xeqy) / len(xeqy) / 2

    print(xeqy)
    num_iter = int(prename.split('-')[-1])
    writer.add_scalar('pr_x_eq_y', xeqy, num_iter)
    return xeqy


if __name__ == '__main__':
    print(pr_to_log(1, sys.argv[1]))
    
    dir_name = 'sex-0617-moco1-init-sex1-x10'
    
    from tensorboardX import SummaryWriter
    writer = SummaryWriter('ckpt-hago/' + dir_name + '/log/val-test-2w')
    
    for k in range(0, 210):
        num_iter = k * 100
        prename = 'ckpt-hago/' + dir_name + '/ret/test-2w-%06d' % num_iter
        pr_to_log(writer, prename)
