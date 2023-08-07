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
sys.path.append('/data/remote/Moco')
sys.path.append('/data/remote/Moco/hago')

import torch.nn as nn
from loader import train_cls_loader, val_cls_loader, uint8_normalize
from loader import train_cls_loader_npy, val_cls_loader_npy
from precise_bn import update_bn_stats
from tensorboardX import SummaryWriter
# from val_hago import validate
# from pr import pr_to_log
from resnest import utils



_FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO, format=_FORMAT, stream=sys.stdout
)

logger = logging.getLogger(__name__)
logger.info('hvd info, size %s, rank %s, local_rank %s.', hvd.size(), hvd.rank(), hvd.local_rank())


from train_hago import load_finetune_checkpoint, get_loader


def parse_args():
    parser = argparse.ArgumentParser()
    # options to ignore (auto added by training system)
    parser.add_argument("--data", type=str)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--hdfs-namenod", type=str)
    
    # Arguments
    parser.add_argument("--list-file", type=str, required=True)
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--load-npy", type=int, default=0)
    parser.add_argument("--donot-shuffle", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--net", type=str, default='resnest50')
    parser.add_argument("--fine-grained", type=int, default=0)
    parser.add_argument("--num-classes", type=str, required=True)
    parser.add_argument("--reduce-dim", type=int, default=0)
    parser.add_argument("--pretrained-ckpt", type=str, default='')
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--fixres", type=int, default=0)
    parser.add_argument("--valloader", type=int, default=0)
    parser.add_argument("--soft-label", type=int, default=0)
    parser.add_argument("--soft-label-gt", type=int, default=0)
    parser.add_argument("--soft-label-t", type=float, default=1.0)
    parser.add_argument("--rand-corner", type=int, default=0)
    parser.add_argument("--loader-keep-wh-ratio", type=int, default=0)
    parser.add_argument("--mixup", type=float, default=0.0)
    
    return parser.parse_args()


from precise_bn import update_bn_stats, get_bn_modules
def calculate_and_update_precise_bn(model, loader, num_iters=4000):
    if hvd.size() > 1:
        num_iters //= hvd.size()
    # Update the bn stats.
    update_bn_stats(model, loader, hvd=hvd, num_iters=num_iters)


def main():
    args = parse_args()
    args.num_classes = [int(i) for i in args.num_classes.split(',')]
    if len(args.num_classes) == 1:
        args.num_classes = args.num_classes[0]
    
    loader = get_loader(args)
    data_size = len(loader)
    logger.info('rank %s, data_size %s', hvd.rank(), data_size)
    
    if args.load_npy == 1:
        normalize_f = lambda x: x.permute(0,3,1,2)
    else:
        normalize_f = uint8_normalize
        
    def _gen_loader():
        while True:
            for data in loader:
                images = data[0]
                images = normalize_f(images.cuda(non_blocking=True))
                yield images
    gen_loader = _gen_loader()
    
    # create model
    if args.net == 'resnet50_x2':
        from resnet_x2 import resnet50
        net_f = resnet50
    elif args.net.startswith('resnest'):
        import resnest
        net_f = resnest.__dict__[args.net]
    elif args.net.startswith('resnet'):
        import torchvision.models as models
        net_f = models.__dict__[args.net]
    
    kwargs = {}
    if args.reduce_dim > 0:
        kwargs['reduce_dim'] = args.reduce_dim
    model = net_f(num_classes=(sum(args.num_classes)
                               if isinstance(args.num_classes, list)
                               else args.num_classes), **kwargs).cuda()
    
    #for i in range(16 + 1):
    #pretrained_ckpt = 'ckpt-yylive/scene_cls-7-fine/ckpt/checkpoint-iter-%03d000.pyth' % i
    
    pretrained_ckpt = args.pretrained_ckpt
    
    if hvd.rank() == 0:
        load_finetune_checkpoint(pretrained_ckpt, model, remove_fc=False)
    if hvd.size() > 1:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    calculate_and_update_precise_bn(model, gen_loader)

    if hvd.rank() == 0:
        checkpoint = {'model_state': model.state_dict()}
        path_to_checkpoint = pretrained_ckpt + '.bn'
        torch.save(checkpoint, path_to_checkpoint)
        logger.info('save checkpoint at %s.', path_to_checkpoint)


if __name__ == "__main__":
    main()

