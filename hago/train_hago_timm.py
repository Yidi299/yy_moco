#!/usr/bin/env python3
import torch
import horovod.torch as hvd
torch.backends.cudnn.benchmark=True

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())
import timm 
import argparse
import sys
import torch
import logging
import time
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import math
import os
sys.path.append('/data1/liuyidi/moco')
sys.path.append('/data1/liuyidi/moco/hago')
from lion_pytorch import Lion
import torch.nn as nn
# from loader_swin import train_cls_loader, val_cls_loader, uint8_normalize
from loader import train_cls_loader, val_cls_loader, uint8_normalize
from loader import train_diff_cls_loader, val_diff_cls_loader
from loader import train_cls_loader_npy, val_cls_loader_npy
from loader import train_cls_loader_npy_img, val_cls_loader_npy_img
from loader import train_loader
# from torchtoolbox.transform import Cutout
from precise_bn import get_bn_modules
from tensorboardX import SummaryWriter
from timm.data.mixup import Mixup
# from val_hago import validate
# from pr import pr_to_log
from resnest import utils
from sam import SAM
from timm.scheduler.cosine_lr import CosineLRScheduler
from swin_transformer_v2 import SwinTransformerV2

#################定义mixup


_FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO, format=_FORMAT, stream=sys.stdout
)

logger = logging.getLogger(__name__)
logger.info('hvd info, size %s, rank %s, local_rank %s.', hvd.size(), hvd.rank(), hvd.local_rank())


from train_self_superv import topks_correct, set_lr, load_last_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    # options to ignore (auto added by training system)
    parser.add_argument("--data", type=str)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--hdfs-namenod", type=str)
    
    # Arguments
    parser.add_argument("--list-file", type=str, default= '/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/train_list_resample_new.txt')
    parser.add_argument("--list-file-map", type=str, default='')
    parser.add_argument("--root-dir", type=str, default= '/data1/liuyidi/scene_cls/6b_2/')
    parser.add_argument("--load-npy", type=int, default=0)
    parser.add_argument("--load-npy-img", type=int, default=0)
    parser.add_argument("--donot-shuffle", type=int, default=0)
    parser.add_argument("--loader-list-repeat-n", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1 )
    parser.add_argument("--net", type=str, default='setting')  ### timm.swinv2_tiny_window8_256
    parser.add_argument("--fine-grained", type=int, default=0)
    parser.add_argument("--fine-grained-n", type=int, default=5)
    parser.add_argument("--fine-grained-w", type=float, default=1.0)
    parser.add_argument("--num-classes", type=str, default=43)
    parser.add_argument("--pretrained-ckpt", type=str, default='/data1/liuyidi/scene_cls/V4.1.2/log_dir/swin_base_22kft1k_1024k/ckpt/checkpoint-iter-1024000.pyth')
    parser.add_argument("--only-fc", type=int, default=0) # optimize only the linear classifier
    parser.add_argument("--base-lr", type=float, default=0.01)  # will be linear scaled by batch-size/256
    parser.add_argument("--lr-stages-step", type=str, default='6000,10000,13000,15000')
    parser.add_argument("--ckpt-log-dir", type=str, default='checkpoints/tmp')
    parser.add_argument("--ckpt-save-interval", type=int, default=1000)
    parser.add_argument("--fp16", type=int, default=0)
    parser.add_argument("--train-epoches", type=int, default=10000)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--final-drop", type=float, default=0.0)
    parser.add_argument("--reduce-dim", type=int, default=0)
    parser.add_argument("--dropblock-prob", type=float, default=0.0)
    parser.add_argument("--no-bias-bn-wd", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--fixres", type=int, default=1)
    parser.add_argument("--valloader", type=int, default=0)
    parser.add_argument("--soft-label", type=int, default=0)
    parser.add_argument("--soft-label-gt", type=int, default=0)
    parser.add_argument("--soft-label-t", type=float, default=1.0)
    parser.add_argument("--sam", type=int, default=1)
    parser.add_argument("--rand-corner", type=int, default=0)
    parser.add_argument("--loader-keep-wh-ratio", type=int, default=0)
    parser.add_argument("--diff-mode", type=int, default=0)
    parser.add_argument("--label_smoothing", type=float, default= 0.1)
    parser.add_argument("--optimizer", type=str, default= 'adamw')
    parser.add_argument("--cos_step", type=int, default= 128000)
    parser.add_argument("--warmup_step", type=int, default= 6400)
    parser.add_argument("--mid", type=int, default= 0)
    parser.add_argument("--accumulation_steps", type=int, default= 0)
    parser.add_argument("--base_batchsize", type=int, default= 1024)
    return parser.parse_args()



def get_lr(base_lr, cur_step, stages_step):
    if len(stages_step) == 0:
        return base_lr
    if cur_step >= stages_step[-1]:
        return -1
    i = 0
    while cur_step > stages_step[i]:
        i += 1
    #if cur_step < 500:
    #    return min(0.01, base_lr * (0.1**i))
    return base_lr * (0.1**i)


def get_w(base_lr, cur_step, stages_step):
    return base_lr ############
    if len(stages_step) == 0:
        return base_lr
    i = 0
    while cur_step > stages_step[i]:
        i += 1
    return base_lr * (0.1**i)


def load_finetune_checkpoint(path_to_checkpoint, model, remove_fc=True, input_dim=3):
    # Load the checkpoint on CPU to avoid GPU mem spike.
    checkpoint = torch.load(path_to_checkpoint, map_location="cpu")
    if 'model_state' not in checkpoint:
        checkpoint = {'model_state': checkpoint}
    if remove_fc and "fc.weight" in checkpoint['model_state']:
        del checkpoint['model_state']["fc.weight"]
        del checkpoint['model_state']["fc.bias"]
    if input_dim > 3:
        assert input_dim % 3 == 0
        checkpoint['model_state']['conv1.0.weight'] = torch.cat(
            [checkpoint['model_state']['conv1.0.weight']] * (input_dim//3), axis=1)
    msg = model.load_state_dict(checkpoint['model_state'], strict=False)
    if remove_fc:
        assert 0 == len(set(msg.missing_keys) - {"fc.weight", "fc.bias",
                                                 "fc2.weight", "fc2.bias"}
                       ), set(msg.missing_keys)
    else:
        assert len(set(msg.missing_keys)) == 0, set(msg.missing_keys)
    logger.info('load_finetune_checkpoint from %s.', path_to_checkpoint)


def load_pretrained(path_to_checkpoint, model, logger):
    checkpoint = torch.load(path_to_checkpoint, map_location='cpu')
    
    state_dict = checkpoint['model_state']
    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    

    del checkpoint
    torch.cuda.empty_cache()


def get_loader(args, threads_gpu=8):
    if args.diff_mode:
        if args.fixres or args.valloader:
            f = val_diff_cls_loader
        else:
            f = train_diff_cls_loader
    elif args.load_npy_img:
        if args.fixres or args.valloader:
            f = val_cls_loader_npy_img
        else:
            f = train_cls_loader_npy_img
    elif args.load_npy:
        if args.fixres or args.valloader:
            f = val_cls_loader_npy
        else:
            f = train_cls_loader_npy
    else:
        if args.fixres or args.valloader:
            f = val_cls_loader
            # f= train_cls_loader
        else:
            f = train_cls_loader
            # f= train_loader
    dataset, loader = f(
                args.list_file, args.root_dir.split('&') if '&' in args.root_dir else args.root_dir,
                list_file_map=args.list_file_map if args.list_file_map != '' else None,
                batch_size=args.batch_size, threads=threads_gpu, hvd=hvd,
                size=args.img_size, shuffle=(args.donot_shuffle==0),
                multi_label=isinstance(args.num_classes, list),
                topk_soft=(args.soft_label==1),
                topk_soft_gt=(args.soft_label_gt==1),
                topk_soft_n=args.num_classes,
                fine_grained=(args.fine_grained==1),
                keep_wh_ratio=(args.loader_keep_wh_ratio==1),
                list_repeat_n=args.loader_list_repeat_n,
                diff_mode=args.diff_mode)
    

    # dataset, loader = f(
    #             args.list_file, args.root_dir.split('&') if '&' in args.root_dir else args.root_dir,
    #             list_file_map=args.list_file_map if args.list_file_map != '' else None,
    #             batch_size=args.batch_size, threads=threads_gpu, hvd=hvd,
    #             size=args.img_size, shuffle=(args.donot_shuffle==0),
    #             )
    return loader


def main():
    args = parse_args()
    # args.num_classes = [int(i) for i in args.num_classes.split(',')]
    # if len(args.num_classes) == 1:
    #     args.num_classes = args.num_classes[0]

    #定义mixup
    mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=43)
    if hvd.rank() == 0:
        if not os.path.exists(args.ckpt_log_dir + '/log/'):
            os.makedirs(args.ckpt_log_dir + '/log/')
        f = open(args.ckpt_log_dir + '/log/para.txt.%s' % int(time.time()), 'w')
        print(args, file=f)
        f.close()
    if args.accumulation_steps > 0:
        base_lr = args.base_lr / args.base_batchsize * args.batch_size * hvd.size()*args.accumulation_steps
    else:
        base_lr = args.base_lr / args.base_batchsize * args.batch_size * hvd.size() # 0.1 for rand init, 0.001~0.00033 for finetune
    lr_stages_step = [int(i) for i in args.lr_stages_step.split(',') if i.strip() != '']
    TENSORBOARD_LOG_DIR = args.ckpt_log_dir + '/log'
    OUTPUT_DIR = args.ckpt_log_dir + '/ckpt'
    #PRETRAINED_PATH = 'checkpoints/ckpt-hago-1/checkpoint-iter-271000.pyth.bn'
    #PRETRAINED_PATH = 'checkpoints/ckpt-hago-1-rmdp/checkpoint-iter-222000.pyth.bn'
    #PRETRAINED_PATH = 'checkpoints/ckpt-imgnet/checkpoint-iter-312000.pyth.bn'
    PRETRAINED_PATH = args.pretrained_ckpt
    CHECKPOINT_PERIOD_STEPS = args.ckpt_save_interval
    all_epoches = args.train_epoches

    loader = get_loader(args)
    data_size = len(loader)
    logger.info('rank %s, data_size %s', hvd.rank(), data_size)
    steps_per_epoch = data_size

    # create model
    if args.net == 'resnet50_x2':
        from resnet_x2 import resnet50
        net_f = resnet50
    elif args.net.startswith('timm'):
        net_f = timm.create_model(args.net.split('.')[-1],num_classes=43)
    elif args.net == 'setting':
        net_f = SwinTransformerV2(img_size= 384,
                                  patch_size= 4 ,
                                  in_chans=3,
                                  num_classes=43,
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
                                  pretrained_window_sizes=[ 16, 16, 16, 8 ])
    
    # print(net_f)
    

    kwargs = {}
    model = net_f.cuda()

    model = model.float().cuda()
    
    normalize_f = lambda x: uint8_normalize(x, input_dim=3 + 3 * args.diff_mode)


    if args.fixres:
        for name, param in model.named_parameters():
            # print(name)
            if not(name in ['head.weight', 'head.bias','norm.weight','norm.bias'])  and 'cpb_mlp' not in name :
                            #'layer4.2.bn3.weight', 'layer4.2.bn3.bias']:
                param.requires_grad = False
            else:
                print(name+'1111111111111111111111')
    if args.mixup > 0:
        criterion =SoftTargetCrossEntropy().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    if args.optimizer == "adamw":
        op =  torch.optim.AdamW
    elif args.optimizer == "Lion":
        op = Lion
    else:
        op =  torch.optim.SGD
    if args.sam == 1:
        optim_f = lambda params, **kwargs: SAM(params, op, rho=0.05, **kwargs)

        
    else:
        optim_f = torch.optim.AdamW

    if op == torch.optim.SGD:
        optimizer = optim_f(model.parameters(),
                            lr=base_lr, momentum=0.9, weight_decay=args.weight_decay)
    elif op == Lion:
        optimizer = optim_f(model.parameters(),
                            lr=base_lr/3, weight_decay=args.weight_decay/3,use_triton=True)
        print("op == Lion")
        print(optimizer)
    elif op == torch.optim.AdamW:
        optimizer = optim_f(model.parameters(),
                            lr=base_lr, eps = 1e-8,betas =(0.9, 0.999) , weight_decay=args.weight_decay)
    if args.fixres:
        parameters = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
        print("参数个数",len(parameters))
        assert len(parameters) >= 4
        optimizer = optim_f([v for n, v in parameters],
                        lr=base_lr, eps = 1e-8,betas =(0.9, 0.999) , weight_decay=args.weight_decay)

    
    if args.sam == 1:
        optimizer_sam = optimizer
        optimizer = optimizer_sam.base_optimizer

    num_steps = args.cos_step
    warmup_steps = args.warmup_step


    ##########注意修改！！！！！！！！
    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=(num_steps - warmup_steps) ,
            lr_min=base_lr*5e-3, 
            warmup_lr_init=base_lr*5e-4,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=True,
        )
    
    if hvd.size() > 1:
        # Add Horovod Distributed Optimizer
        if args.sam == 1:
            optimizer = hvd.DistributedOptimizer(optimizer,
                        named_parameters=model.named_parameters(),
                        backward_passes_per_step=args.accumulation_steps *2 if args.accumulation_steps >0 else 2)
            optimizer_sam.base_optimizer = optimizer
        else:
            optimizer = hvd.DistributedOptimizer(optimizer,
                        named_parameters=model.named_parameters(),
                        backward_passes_per_step=args.accumulation_steps *2 if args.accumulation_steps >0 else 2)
            
    total_iter = 0
    if hvd.rank() == 0:
        epoch = load_last_checkpoint(OUTPUT_DIR, model, optimizer)
        if epoch:
            total_iter = int(epoch * steps_per_epoch + 0.5)
            #total_iter = 50000 # ==========================================================================================
        elif PRETRAINED_PATH:
            if args.fixres or args.mid:
                # load_finetune_checkpoint(PRETRAINED_PATH, model, remove_fc=False)
                load_pretrained(PRETRAINED_PATH, model, logger)
            else:
                weights_dict = torch.load(PRETRAINED_PATH, map_location='cpu')["model"]
                # 删除有关分类类别的权重
                for k in list(weights_dict.keys()):
                    if "head" in k:
                        del weights_dict[k]
                print(model.load_state_dict(weights_dict, strict=False))


    if hvd.size() > 1:
        # Broadcast parameters from rank 0 to all other processes.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        total_iter = hvd.broadcast_object(total_iter, root_rank=0)
    ckp_save_iter = total_iter

    if hvd.rank() == 0:
        writer = SummaryWriter(TENSORBOARD_LOG_DIR)
        # writer2 = SummaryWriter(TENSORBOARD_LOG_DIR + '/val-test-2w')
    
    
    model.train()
    t0 = time.time()
    logger.info('rank %s, total_iter %s', hvd.rank(), total_iter)


    images_ls ,target_ls = [],[]

    for epoch in range(all_epoches):
        for cur_iter, data in enumerate(loader):
            images, target = data
            # images = torch.stack(images,dim =0)
            # print(images.shape)
            images = normalize_f(images.cuda(non_blocking=True))
            target = target.cuda(non_blocking=True)


            if args.mixup > 0:
                images, target = mixup_fn(images, target)
            target0 = target

            cur_epoch = total_iter / steps_per_epoch
            if cur_epoch > all_epoches:
                break
            
            lr = get_lr(base_lr, total_iter, lr_stages_step)
            # if lr < 0:
            #     break
            # set_lr(optimizer, lr)
            # lr_scheduler.step_update((total_iter))
            lr = lr_scheduler._get_lr(total_iter)[0]
            set_lr(optimizer, lr)
            images = images.contiguous()
            output = model(images)
            if total_iter > args.cos_step:
                break
            
            loss = criterion(output, target)
            if math.isnan(loss):
                raise RuntimeError("ERROR: Got NaN losses")
            
            if args.accumulation_steps:
                
                
                if args.sam == 1:
                    # first forward-backward pass

                    
                    loss = loss / args.accumulation_steps
                    
                    loss.backward()

                    ##暂存数据第二步计算
                    images_ls.append(images)
                    target_ls.append(target)

                    if (total_iter + 1) % args.accumulation_steps == 0 or (total_iter + 1 == len(loader)):
                        optimizer_sam.first_step(zero_grad=True)
                        optimizer.synchronize()
                        for i in range(len(images_ls)):
                            output2 = model(images_ls[i])
                            loss2 = criterion(output2, target_ls[i])/args.accumulation_steps
                            if math.isnan(loss2):
                                raise RuntimeError("ERROR: Got NaN loss2")
                            loss2.backward()
                        with optimizer.skip_synchronize():
                            optimizer_sam.second_step(zero_grad=True) 
                        
                        ##清空暂存数据
                        images_ls ,target_ls = [],[]  

                        
                else :
                    loss = loss / args.accumulation_steps
                    loss.backward()

                    if (cur_iter + 1) % args.accumulation_steps == 0 or (cur_iter + 1 == len(loader)):
                        optimizer.step()
                        optimizer.zero_grad()

                       
                        
                        
                
            else:

                if args.sam == 1:
                    
                    # first forward-backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # optimizer.synchronize()
                    optimizer_sam.first_step(zero_grad=True)
                    # second forward-backward pass
                    output2 = model(images)
                    loss2 = criterion(output2, target)
                    if math.isnan(loss2):
                        raise RuntimeError("ERROR: Got NaN loss2")
                    loss2.backward()
                    optimizer.synchronize()
                    with optimizer.skip_synchronize():
                        optimizer_sam.second_step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if args.mixup > 0 or args.soft_label == 1:
                target0 = torch.argmax(target0, dim=-1)
            
            num_topks_correct = topks_correct(output, target0, [1])
            
            top1_acc = num_topks_correct[0] / output.size(0)
            if hvd.size() > 1:
                loss = hvd.allreduce(loss)
                top1_acc = hvd.allreduce(top1_acc)

            if hvd.rank() == 0:
                t = time.time()
                lr = optimizer.param_groups[0]['lr']
                logger.info('epoch %.6f, iter %s, loss %.6f, top1_acc %.6f, lr %.6f, step time %.6f',
                            cur_epoch, total_iter, loss, top1_acc, lr, t-t0)
                writer.add_scalar('1-top1_acc', top1_acc, total_iter)
                
                writer.add_scalar('2-loss', loss, total_iter)
                writer.add_scalar('3-lr', lr, total_iter)
                writer.add_scalar('4-steps_per_s', 1.0 / (t-t0), total_iter)
                t0 = t

            #--------------save ckpt
            if total_iter == 0 or total_iter - ckp_save_iter >= CHECKPOINT_PERIOD_STEPS:
                ckp_save_iter = total_iter
                path_to_checkpoint = OUTPUT_DIR + '/' + 'checkpoint-iter-%06d.pyth' % total_iter
                if hvd.rank() == 0:
                    if not os.path.exists(OUTPUT_DIR):
                        os.makedirs(OUTPUT_DIR)
                    checkpoint = {
                        "epoch": cur_epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "hvd.size": hvd.size(),
                        "args": vars(args),}
                    torch.save(checkpoint, path_to_checkpoint)
                    logger.info('save checkpoint at step %s.', total_iter)

            total_iter += 1
        # if lr < 0:
        #     break
        loader = get_loader(args) # shuffle data

    #--------------save ckpt
    path_to_checkpoint = OUTPUT_DIR + '/' + 'checkpoint-iter-%06d.pyth' % total_iter
    if total_iter - ckp_save_iter > 1 and hvd.rank() == 0:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        checkpoint = {
            "epoch": cur_epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "hvd.size": hvd.size(),
            "args": vars(args),}
        torch.save(checkpoint, path_to_checkpoint)
        logger.info('save checkpoint at step %s.', total_iter)


if __name__ == "__main__":
    main()

