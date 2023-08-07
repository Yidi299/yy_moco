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
from transformer_block import ImageTransformer,ImageTransformer2
from Lstm_block import TimeSeriesClassifier
from loader import val_cls_loader, uint8_normalize
from loader import val_diff_cls_loader
from loader import val_cls_loader_npy
from loader import val_cls_loader_npy_img
from loader_my import val_cls_loader_4
from tensorboardX import SummaryWriter

from swin_transformer_v2 import SwinTransformerV2


_FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO, format=_FORMAT, stream=sys.stdout
)

logger = logging.getLogger(__name__)
# logger.info('hvd info, size %s, rank %s, local_rank %s.', hvd.size(), hvd.rank(), hvd.local_rank())


from train_self_superv import load_last_checkpoint
from train_hago_4 import load_finetune_checkpoint

# from train_hago_4 import CustomModel , CustomModel2 , CustomModel3

from model_test import CustomModel3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-file", type=str,default= '/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/val*4_list.txt')
    parser.add_argument("--root-dir", type=str,default= '/data1/liuyidi/scene_cls/6b_2/')
    parser.add_argument("--load-npy", type=int, default=0)
    parser.add_argument("--val_4", type=int, default=1)
    parser.add_argument("--load-npy-img", type=int, default=0)
    parser.add_argument("--diff-mode", type=int, default=0)
    parser.add_argument("--num-classes", type=int,  default='43')
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--net", type=str, default='resnest50')
    parser.add_argument("--reduce-dim", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default='/data1/liuyidi/scene_cls/V4.1/log_dir/V4.1_test27_1_fix/ckpt/checkpoint-iter-008000.pyth')
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
        f = val_cls_loader_4
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
        normalize_f = lambda x: uint8_normalize(x, input_dim=3 + 3 * args.diff_mode)

    data = []
    data_part_i = 0
    for cur_iter, (images, target) in enumerate(loader):
        # output_ls = []
        # images_ls = torch.unbind(images_ls, dim=1)
        # for images in images_ls:


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
        
        if hvd.size() > 1:
            output = hvd.allgather(output)
            target = hvd.allgather(target)
            
        
        
                
        if hvd.rank() == 0:
            output = output.cpu().float().numpy()
            target = target.cpu().float().numpy()
            # print(output.shape, output.dtype, target.shape, target.dtype)
            data.append([output.astype('float32'), target.astype('int32')])
            # print(len(data))
            cur_epoch = total_iter / data_size
            t = time.time()
            logger.info('epoch %.6f, iter %s, step time %.6f',
                        cur_epoch, total_iter, t-t0)
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
    elif  args.net == 'setting':
        net_f = SwinTransformerV2(img_size= 384,
                                  patch_size= 4 ,
                                  in_chans=3,
                                  num_classes=0,
                                  embed_dim= 128 ,
                                  depths= [ 2, 2, 18, 2 ],
                                  num_heads= [ 4, 8, 16, 32 ] ,
                                  window_size=24,
                                  mlp_ratio=4. ,
                                  qkv_bias=True,
                                  drop_rate=0.0,
                                  drop_path_rate= 0.3,
                                  ape=False,
                                  patch_norm=True,
                                  use_checkpoint=False,
                                  pretrained_window_sizes=[ 16, 16, 16, 8 ]  )

    kwargs = {}
    if args.reduce_dim > 0:
        kwargs['reduce_dim'] = args.reduce_dim
    if args.diff_mode:
        kwargs['input_dim'] = 3 + 3 * args.diff_mode
    # model0 = net_f(num_classes=num_cls, **kwargs).cuda()#.half()
    model0 = net_f
    if args.fp16:
        model = model.half()


    # if hvd.rank() == 0:
    #     mk = torch.load('/data1/liuyidi/scene_cls/V4.1/log_dir/V4.1_test27_1_fix/ckpt/checkpoint-iter-008000.pyth', map_location="cpu")
    #     ck  = mk['model_state']
    #     msg = model0.load_state_dict(ck, strict=False)


    # model0 = torch.nn.Sequential(*list(model0.children())[:-1])

    # if hvd.rank() == 0:
    #     ck0 = torch.load('/data1/liuyidi/scene_cls/V4.1/log_dir/V4.1_test27_1_fix/ckpt/checkpoint-iter-008000.pyth', map_location="cpu")['model_state']
    #     msg0 = model0.load_state_dict(ck0, strict=False)
    #     model0 = torch.nn.Sequential( *( list(model.children())[:] ) )

    # model = CustomModel3(model0, args.num_classes,input_features=2048).cuda()
    model = ImageTransformer2(model0,d_model=1024,dropout_rate=0.6).cuda()
    # model = TimeSeriesClassifier(model0).cuda()
    
    model.eval()

    
    

    
            
        

    
        
    #     mk = torch.load('/data1/liuyidi/scene_cls/V4.1/log_dir/V4.1_test27_1_fix/ckpt/checkpoint-iter-008000.pyth', map_location="cpu")
    #     ck  = mk['model_state']
    #     for name, param in ck.copy().items(): ##字典同时遍历和修改会报错，因此在副本上进行遍历
    #         ck ['pretrained_model.'+name] =  ck.pop(name)
    #     msg = model.load_state_dict(ck, strict=False)

          

    
   
    
   
    
    


    if hvd.rank() == 0:
        if num_cls == 0: # 输出fc前面的feature
            load_finetune_checkpoint(ckpname, model, remove_fc=True)
        else:
            load_last_checkpoint(OUTPUT_DIR, model, name=CKPT_NAME)
        
        #修改权重
        # cls_weight = torch.load('/data1/liuyidi/scene_cls/V4.1/log_dir/V4.1_test27_1_fix/ckpt/checkpoint-iter-008000.pyth', map_location="cpu")
        # # #将cls_weight最后一层的参数赋值给model的最后一层
        # model.fc.weight.data = cls_weight['model_state']['fc.weight'].data.cuda()
        # model.fc.bias.data = cls_weight['model_state']['fc.bias'].data.cuda()
    

    #     ck_all = torch.load('/data1/liuyidi/scene_cls/V4.1/log_dir/V4.2_trans_try/ckpt/checkpoint-iter-002000.pyth', map_location="cpu")['model_state']
    #     msg = model.load_state_dict(ck_all, strict=False)

    #     pre = torch.load('/data1/liuyidi/scene_cls/V4.1/log_dir/V4.1_test27_1_fix/ckpt/checkpoint-iter-008000.pyth', map_location="cuda")['model_state']

    #     for i,j in zip(pre.keys(),model.feature_extractor.state_dict().keys()):
    #         if torch.equal(model.feature_extractor.state_dict()[j].data,pre[i].data): ##判断两个tensor是否相等使用torch.equal
    #             print(i,'same')
    #         else:
    #             print(i,'not same')
    
        


    if hvd.size() > 1:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    loader = get_loader()
    validate(model, loader, out_name)


if __name__ == "__main__":
    main()

