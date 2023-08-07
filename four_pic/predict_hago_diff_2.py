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
import numpy as np
sys.path.append('/data1/liuyidi/moco')
sys.path.append('/data1/liuyidi/moco/hago')

import torch.nn as nn
from loader import val_cls_loader, uint8_normalize
from loader import val_diff_cls_loader
from loader import val_cls_loader_npy
from loader import val_cls_loader_npy_img
from loader_my import val_cls_loader_4
from tensorboardX import SummaryWriter


_FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO, format=_FORMAT, stream=sys.stdout
)

logger = logging.getLogger(__name__)
# logger.info('hvd info, size %s, rank %s, local_rank %s.', hvd.size(), hvd.rank(), hvd.local_rank())


from train_self_superv import load_last_checkpoint
from train_hago_4_2 import load_finetune_checkpoint
from loader_my_diff import val_cls_loader_frame_diff,uint8_normalize_diff


def parse_args():
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--list-file", type=str,default= '/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/val*4_list.txt')
    parser.add_argument("--root-dir", type=str,default= '/data1/liuyidi/scene_cls/6b_2/')
    parser.add_argument("--load-npy", type=int, default=0)
    parser.add_argument("--val_4", type=int, default=1)
    parser.add_argument("--load-npy-img", type=int, default=0)
    parser.add_argument("--diff-mode", type=int, default=0)
    parser.add_argument("--num-classes", type=int,  default='43')
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--net", type=str, default='resnest50')
    parser.add_argument("--reduce-dim", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default='/home/pengguozhu/liuyidi/logdir/V4.1_4/diff_try0/ckpt/checkpoint-iter-256000.pyth')
    parser.add_argument("--out", type=str, default='/data1/liuyidi/scene_cls/V4.1/log_dir/V4.2_linear/val_debug' )
    parser.add_argument("--out-per-n", type=int, default=0)
    parser.add_argument("--fp16", type=int, default=0)
    return parser.parse_args()
args = parse_args()


def get_loader():
    if args.diff_mode:
        f = val_diff_cls_loader
    elif args.load_npy_img == 1:
        f = val_cls_loader_npy_img
    elif args.load_npy == 1:
        f = val_cls_loader_npy
    elif args.val_4 == 1:
        f = val_cls_loader_frame_diff
    else:
        f = val_cls_loader
    dataset, loader = f(
        args.list_file, args.root_dir.split('&') if '&' in args.root_dir else args.root_dir,
        batch_size=args.batch_size, threads=32, hvd=hvd, line_num_mode=True,
        size=args.img_size, shuffle=False, diff_mode=args.diff_mode)
    return loader


def validate(model, loader, OUT_NAME=None):
    data_size = len(loader)
    t0 = time.time()
    
    if hvd.rank() == 0:
        logger.info('rank %s, size %s, data_size %s', hvd.rank(), hvd.size(), data_size)
    total_iter = 0

    if args.load_npy_img == 1:
        normalize_f = [lambda x: (x.permute(0,3,1,2) if len(x.shape)==4
                                     else x.permute(0,1,4,2,3)),
                       lambda x: uint8_normalize(x, input_dim=3 + 3 * args.diff_mode)]
    elif args.load_npy == 1:
        normalize_f = lambda x: (x.permute(0,3,1,2) if len(x.shape)==4
                                 else x.permute(0,1,4,2,3))
    else:
        normalize_f = lambda x: uint8_normalize_diff(x, input_dim=4)

    data = []
    data_part_i = 0
    for cur_iter, (images_ls, target) in enumerate(loader):
        output_ls = []
        images = images_ls.reshape(-1, 4, 256, 256)
        
        if isinstance(normalize_f, list):
            assert isinstance(images, list) and len(images) == len(normalize_f)
            images = [normf(img.cuda(non_blocking=True)) for img, normf in zip(images, normalize_f)]
        else:
            images = normalize_f(images.cuda(non_blocking=True))#.half()
        if args.fp16:
            images = images.half()
        if isinstance(normalize_f, list):
            images = [img.contiguous() for img in images]
        else:
            images = images.contiguous()

        with torch.no_grad():
            output = model(images)
            output = output.reshape(-1, 4, args.num_classes)
            output = torch.mean(output, dim=1)
        
        if hvd.size() > 1:
            output = hvd.allgather(output)
            target = hvd.allgather(target)
                
        if hvd.rank() == 0:
            output = output.cpu().float().numpy()
            target = target.cpu().float().numpy()
            print(output.shape, output.dtype, target.shape, target.dtype)
            data.append([output.astype('float32'), target.astype('int32')])
            print(len(data))
            cur_epoch = total_iter / data_size
            t = time.time()
            # logger.info('epoch %.6f, iter %s, step time %.6f',
            #             # cur_epoch, total_iter, t-t0)
            t0 = t
            total_iter += 1
            if args.out_per_n > 0 and len(data) >= args.out_per_n:
                if os.path.dirname(OUT_NAME) !='' and not os.path.exists(os.path.dirname(OUT_NAME)):
                    os.makedirs(os.path.dirname(OUT_NAME))
                np.save(OUT_NAME + '-%s-pr.npy' % data_part_i, np.concatenate([i[0] for i in data]))
                np.save(OUT_NAME + '-%s-lb.npy' % data_part_i, np.concatenate([i[1] for i in data]))
                data = []
                data_part_i += 1
    if hvd.rank() == 0 and len(data) > 0:
        if os.path.dirname(OUT_NAME) !='' and not os.path.exists(os.path.dirname(OUT_NAME)):
            os.makedirs(os.path.dirname(OUT_NAME))
        np.save(OUT_NAME + '-pr.npy', np.concatenate([i[0] for i in data]))
        np.save(OUT_NAME + '-lb.npy', np.concatenate([i[1] for i in data]))
        print(len(data))
        # data0 = np.array(data)
        # np.save('data0',data0)



def main():
    ckpname = args.ckpt
    tokens = ckpname.split('/')
    OUTPUT_DIR = '/'.join(tokens[:-1])
    CKPT_NAME = tokens[-1]
    num_cls = args.num_classes
    out_name = args.out

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
    elif args.net.startswith('CnnHead'):
        if args.net.split('.')[0] == 'CnnHead':
            from seq_cls import CnnHead
            cf = CnnHead
        elif args.net.split('.')[0] == 'CnnHead2':
            from seq_cls import CnnHead2
            cf = CnnHead2
        if args.net.split('.')[-1].startswith('resnest'):
            import resnest
            def net_f(**kwargs):
                return cf(arch=resnest.__dict__[args.net.split('.')[-1]], **kwargs)
        elif args.net.split('.')[-1].startswith('resnet'):
            import torchvision.models as models
            def net_f(**kwargs):
                return cf(arch=models.__dict__[args.net.split('.')[-1]], **kwargs)

    kwargs = {}
    if args.reduce_dim > 0:
        kwargs['reduce_dim'] = args.reduce_dim
    if args.diff_mode:
        kwargs['input_dim'] = 3 + 3 * args.diff_mode
    model = net_f(num_classes=num_cls,input_dim = 4 ,**kwargs).cuda()#.half()
    if args.fp16:
        model = model.half()
    model.eval()

    if hvd.rank() == 0:
        if num_cls == 0: # 输出fc前面的feature
            load_finetune_checkpoint(ckpname, model, remove_fc=True)
        else:
            load_last_checkpoint(OUTPUT_DIR, model, name=CKPT_NAME)
    if hvd.size() > 1:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    loader = get_loader()
    validate(model, loader, out_name)


if __name__ == "__main__":
    main()

