#!/usr/bin/env python3
import torch
import horovod.torch as hvd
torch.backends.cudnn.benchmark=True

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())

import argparse
import sys
import torch
import logging
import time
import math
import os

import torch.nn as nn
from loader import val_cls_loader, uint8_normalize
from tensorboardX import SummaryWriter


_FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO, format=_FORMAT, stream=sys.stdout
)

logger = logging.getLogger(__name__)
logger.info('hvd info, size %s, rank %s, local_rank %s.', hvd.size(), hvd.rank(), hvd.local_rank())


from train_self_superv import parse_args, topks_correct


def load_last_checkpoint(dir_to_checkpoint, model, name=None):
    if name is None:
        names = os.listdir(dir_to_checkpoint) if os.path.exists(dir_to_checkpoint) else []
        names = [f for f in names if "checkpoint" in f]
        if len(names) == 0:
            return None
        name = sorted(names)[-1]
    path_to_checkpoint = os.path.join(dir_to_checkpoint, name)
    # Load the checkpoint on CPU to avoid GPU mem spike.
    checkpoint = torch.load(path_to_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    ckp_step = int(name.split('.')[0].split('-')[-1])
    logger.info('checkpoint loaded from %s (ckp_step %s).', path_to_checkpoint, ckp_step)
    return ckp_step


def get_loader(batch_size):
    dataset, loader = val_cls_loader(
        'data/val.txt', 'http://filer.ai.yy.com:9889/dataset/heliangliang/imagenet/val/',
        batch_size=batch_size, threads=32, hvd=hvd)
    return loader


def main():
    args = parse_args()
    batch_size = 64
    #TENSORBOARD_LOG_DIR = './checkpoints/log-fc-1/val'
    #OUTPUT_DIR          = './checkpoints/ckpt-fc-1'
    TENSORBOARD_LOG_DIR = './ckpt-byol/imagenet-lr-0.1/log-fc/val'
    OUTPUT_DIR          = './ckpt-byol/imagenet-lr-0.1/ckpt-fc'

    loader = get_loader(batch_size)

    import torchvision.models as models
    model = models.__dict__['resnet50']().cuda()
    #from resnet_x2 import resnet50
    #model = resnet50(num_classes=1000).cuda()

    model.eval()
    data_size = len(loader)
    t0 = time.time()

    logger.info('rank %s, data_size %s', hvd.rank(), data_size)
    total_iter = 0

    if hvd.rank() == 0:
        ckp_step = load_last_checkpoint(OUTPUT_DIR, model)
    if hvd.size() > 1:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    logger.info('rank %s, total_iter %s', hvd.rank(), total_iter)
    top1_acc_all = []
    for cur_iter, (images, target) in enumerate(loader):
        images = uint8_normalize(images.cuda(non_blocking=True))
        target = target.cuda(non_blocking=True)
        
        with torch.no_grad():
            output = model(images)
            num_topks_correct = topks_correct(output, target, [1])
            top1_acc = num_topks_correct[0] / output.size(0)
        if hvd.size() > 1:
            top1_acc = hvd.allreduce(top1_acc)

        cur_epoch = total_iter / data_size
        top1_acc_all.append(top1_acc)

        if hvd.rank() == 0:
            t = time.time()
            logger.info('epoch %.6f, iter %s, top1_acc %.6f (%.6f), step time %.6f',
                        cur_epoch, total_iter, top1_acc, sum(top1_acc_all)/len(top1_acc_all), t-t0)
            t0 = t
        total_iter += 1

    if hvd.rank() == 0:
        writer = SummaryWriter(TENSORBOARD_LOG_DIR)
        writer.add_scalar('1-top1_acc', sum(top1_acc_all)/len(top1_acc_all), ckp_step)


if __name__ == "__main__":
    main()

