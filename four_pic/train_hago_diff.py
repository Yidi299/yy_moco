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
sys.path.append('/data1/liuyidi/moco')
sys.path.append('/data1/liuyidi/moco/hago')
import timm
import torch.nn as nn
from loader import train_cls_loader, val_cls_loader, uint8_normalize
from loader import train_diff_cls_loader, val_diff_cls_loader
from loader import train_cls_loader_npy, val_cls_loader_npy
from loader import train_cls_loader_npy_img, val_cls_loader_npy_img
from loader_my_diff import train_cls_loader_frame_diff,uint8_normalize_diff
from precise_bn import get_bn_modules
from tensorboardX import SummaryWriter
# from val_hago import validate
# from pr import pr_to_log
from resnest import utils
from sam import SAM


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
    parser.add_argument("--list-file", type=str, default='/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/data*4_1_list.txt')
    parser.add_argument("--list-file-map", type=str, default='')
    parser.add_argument("--root-dir", type=str, default='/data1/liuyidi/scene_cls/6b_2/')
    parser.add_argument("--load-npy", type=int, default=0)
    parser.add_argument("--load-npy-img", type=int, default=0)
    parser.add_argument("--donot-shuffle", type=int, default=0)
    parser.add_argument("--loader-list-repeat-n", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--net", type=str, default='resnest50')
    parser.add_argument("--fine-grained", type=int, default=0)
    parser.add_argument("--fine-grained-n", type=int, default=5)
    parser.add_argument("--fine-grained-w", type=float, default=1.0)
    parser.add_argument("--num-classes", type=str, default=43)
    parser.add_argument("--pretrained-ckpt", type=str)
    parser.add_argument("--only-fc", type=int, default=0) # optimize only the linear classifier
    parser.add_argument("--base-lr", type=float, default=0.01)  # will be linear scaled by batch-size/256
    parser.add_argument("--lr-stages-step", type=str, default='4000,6000,8001')
    parser.add_argument("--ckpt-log-dir", type=str, default='/data1/liuyidi/scene_cls/V4.1/log_dir/V4.2_linear')
    parser.add_argument("--ckpt-save-interval", type=int, default=1000)
    parser.add_argument("--fp16", type=int, default=0)
    parser.add_argument("--train-epoches", type=int, default=10000)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--final-drop", type=float, default=0.0)
    parser.add_argument("--reduce-dim", type=int, default=0)
    parser.add_argument("--dropblock-prob", type=float, default=0.0)
    parser.add_argument("--no-bias-bn-wd", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--fixres", type=int, default=0)
    parser.add_argument("--valloader", type=int, default=0)
    parser.add_argument("--soft-label", type=int, default=0)
    parser.add_argument("--soft-label-gt", type=int, default=0)
    parser.add_argument("--soft-label-t", type=float, default=1.0)
    parser.add_argument("--sam", type=int, default=1)
    parser.add_argument("--rand-corner", type=int, default=0)
    parser.add_argument("--loader-keep-wh-ratio", type=int, default=0)
    parser.add_argument("--diff-mode", type=int, default=0)
    
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
    if input_dim == 4 :
        # del checkpoint['model_state']["fc.weight"]
        # del checkpoint['model_state']["fc.bias"]
        del checkpoint['model_state']["conv1.0.weight"]
        
    # if input_dim > 3:
    #     assert input_dim % 3 == 0
    #     checkpoint['model_state']['conv1.0.weight'] = torch.cat(
    #         [checkpoint['model_state']['conv1.0.weight']] * (input_dim//3), axis=1)
    msg = model.load_state_dict(checkpoint['model_state'], strict=False)
    # if remove_fc:
    #     assert 0 == len(set(msg.missing_keys) - {"fc.weight", "fc.bias",
    #                                              "fc2.weight", "fc2.bias"}
    #                    ), set(msg.missing_keys)
    # else:
    #     assert len(set(msg.missing_keys)) == 0, set(msg.missing_keys)
    logger.info('load_finetune_checkpoint from %s.', path_to_checkpoint)


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
        else:
            f = train_cls_loader_frame_diff
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
                randcorner=(args.rand_corner==1),
                keep_wh_ratio=(args.loader_keep_wh_ratio==1),
                list_repeat_n=args.loader_list_repeat_n,
                diff_mode=args.diff_mode)
    return loader


def main():
    args = parse_args()
    # args.num_classes = [int(i) for i in args.num_classes.split(',')]
    args.num_classes = 43
    # if len(args.num_classes) == 1:
    #     args.num_classes = args.num_classes[0]
    if hvd.rank() == 0:
        if not os.path.exists(args.ckpt_log_dir + '/log/'):
            os.makedirs(args.ckpt_log_dir + '/log/')
        f = open(args.ckpt_log_dir + '/log/para.txt.%s' % int(time.time()), 'w')
        print(args, file=f)
        f.close()
    base_lr = args.base_lr / 256 * args.batch_size * hvd.size() # 0.1 for rand init, 0.001~0.00033 for finetune
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
    if args.final_drop > 0.0:
        kwargs['final_drop'] = args.final_drop
    if args.reduce_dim > 0:
        kwargs['reduce_dim'] = args.reduce_dim
    if args.diff_mode:
        kwargs['input_dim'] = 3 + 3 * args.diff_mode
    if args.fine_grained == 1:
        assert not isinstance(args.num_classes, list)
        model = net_f(num_classes=args.num_classes+args.fine_grained_n, **kwargs).cuda()
        criterion_fine = [nn.BCEWithLogitsLoss(reduction='none').cuda()
                          for i in range(args.fine_grained_n)]
    else:
        model = net_f(num_classes=(sum(args.num_classes)
                                   if isinstance(args.num_classes, list)
                                   else args.num_classes),input_dim = 4, **kwargs).cuda()

    if args.only_fc:
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        # init the fc layer
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
    elif args.fixres:
        # freeze all layers but the last fc and last bn
        for name, param in model.named_parameters():
            if not(name in ['fc.weight', 'fc.bias',] or name.startswith('layer4')):
                            #'layer4.2.bn3.weight', 'layer4.2.bn3.bias']:
                param.requires_grad = False

    if args.fp16 == 1:
        model = model.half().cuda()
        for bn in get_bn_modules(model):
            bn.float()
        if args.load_npy_img == 1:
            normalize_f = [lambda x: (x.type(torch.half).permute(0,3,1,2) if len(x.shape)==4
                                     else x.type(torch.half).permute(0,1,4,2,3)),
                           lambda x: uint8_normalize(x, dtype=torch.half,
                                                     input_dim=3 + 3 * args.diff_mode)]
        elif args.load_npy == 1:
            normalize_f = lambda x: (x.type(torch.half).permute(0,3,1,2) if len(x.shape)==4
                                     else x.type(torch.half).permute(0,1,4,2,3))
        else:
            normalize_f = lambda x: uint8_normalize(x, dtype=torch.half,
                                                    input_dim=3 + 3 * args.diff_mode)
    else:
        model = model.float().cuda()
        if args.load_npy_img == 1:
            normalize_f = [lambda x: (x.permute(0,3,1,2) if len(x.shape)==4
                                     else x.permute(0,1,4,2,3)),
                           lambda x: uint8_normalize(x, input_dim=3 + 3 * args.diff_mode)]
        elif args.load_npy == 1:
            normalize_f = lambda x: (x.permute(0,3,1,2) if len(x.shape)==4
                                     else x.permute(0,1,4,2,3))
        else:
            normalize_f = lambda x: uint8_normalize_diff(x, input_dim=4)
    
    if args.mixup > 0:
        if isinstance(args.num_classes, list):
            criterion = [utils.NLLMultiLabelSmooth().cuda()
                         for j in args.num_classes]
        else:
            criterion = utils.NLLMultiLabelSmooth().cuda()
    elif args.soft_label == 1:
        if isinstance(args.num_classes, list):
            criterion = [utils.SoftCrossEntropyLoss(t=args.soft_label_t).cuda()
                         for j in args.num_classes]
        else:
            criterion = utils.SoftCrossEntropyLoss(t=args.soft_label_t).cuda()
    elif args.soft_label_gt == 1:
        assert not isinstance(args.num_classes, list)
        criterion = utils.SoftCrossEntropyLoss(t=args.soft_label_t).cuda()
        criterion2 = nn.CrossEntropyLoss().cuda()
    else:
        if isinstance(args.num_classes, list):
            criterion = [nn.CrossEntropyLoss().cuda()
                         for j in args.num_classes]
        else:
            criterion = nn.CrossEntropyLoss().cuda()
    
    if args.sam == 1:
        optim_f = lambda params, **kwargs: SAM(params, torch.optim.SGD, rho=0.05, **kwargs)
    else:
        optim_f = torch.optim.SGD

    if args.only_fc:
        # optimize only the linear classifier
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias
        optimizer = optim_f(parameters, lr=base_lr, momentum=0.9, weight_decay=0.0)
    elif args.fixres:
        parameters = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
        assert len(parameters) >= 4
        # optimizer = optim_f(parameters, lr=base_lr, momentum=0.9, weight_decay=0.0)
        if args.no_bias_bn_wd == 1:
            bn_params = [v for n, v in parameters if ('bn' in n or 'bias' in n)]
            rest_params = [v for n, v in parameters if not ('bn' in n or 'bias' in n)]
            optimizer = optim_f([{'params': bn_params, 'weight_decay': 0 },
                                 {'params': rest_params, 'weight_decay': args.weight_decay}],
                                lr=base_lr, momentum=0.9, weight_decay=args.weight_decay)
        else:
            optimizer = optim_f([v for n, v in parameters],
                                lr=base_lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        if args.no_bias_bn_wd == 1:
            bn_params = [v for n, v in model.named_parameters() if ('bn' in n or 'bias' in n)]
            rest_params = [v for n, v in model.named_parameters() if not ('bn' in n or 'bias' in n)]
            optimizer = optim_f([{'params': bn_params, 'weight_decay': 0 },
                                 {'params': rest_params, 'weight_decay': args.weight_decay}],
                                lr=base_lr, momentum=0.9, weight_decay=args.weight_decay)
        else:
            optimizer = optim_f(model.parameters(),
                                lr=base_lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.sam == 1:
        optimizer_sam = optimizer
        optimizer = optimizer_sam.base_optimizer

    if hvd.size() > 1:
        # Add Horovod Distributed Optimizer
        if args.sam == 1:
            optimizer = hvd.DistributedOptimizer(optimizer,
                        named_parameters=model.named_parameters(),
                        backward_passes_per_step=2)
            optimizer_sam.base_optimizer = optimizer
        else:
            optimizer = hvd.DistributedOptimizer(optimizer,
                        named_parameters=model.named_parameters())

    total_iter = 0
    if hvd.rank() == 0:
        epoch = load_last_checkpoint(OUTPUT_DIR, model, optimizer)
        if epoch:
            total_iter = int(epoch * steps_per_epoch + 0.5)
            #total_iter = 50000 # ==========================================================================================
        elif PRETRAINED_PATH:
            #load_finetune_checkpoint(PRETRAINED_PATH, model, remove_fc=(False if args.fixres else True))
            try:
                if args.net.startswith('CnnHead2.'):
                    load_finetune_checkpoint(PRETRAINED_PATH.split('&')[0], model.arch, remove_fc=False)
                    load_finetune_checkpoint(PRETRAINED_PATH.split('&')[1], model.arch2, remove_fc=False)
                elif args.net.startswith('CnnHead.'):
                    load_finetune_checkpoint(PRETRAINED_PATH, model.arch, remove_fc=False)
                else:
                    load_finetune_checkpoint(PRETRAINED_PATH, model, remove_fc=True, input_dim=4)
            except:
                if args.net.startswith('CnnHead2.'):
                    load_finetune_checkpoint(PRETRAINED_PATH.split('&')[0], model.arch, remove_fc=True)
                    load_finetune_checkpoint(PRETRAINED_PATH.split('&')[1], model.arch2, remove_fc=True)
                elif args.net.startswith('CnnHead.'):
                    load_finetune_checkpoint(PRETRAINED_PATH, model.arch, remove_fc=True)
                else:
                    load_finetune_checkpoint(PRETRAINED_PATH, model, remove_fc=True, input_dim=4)
    if hvd.size() > 1:
        # Broadcast parameters from rank 0 to all other processes.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        total_iter = hvd.broadcast_object(total_iter, root_rank=0)
    ckp_save_iter = total_iter

    if hvd.rank() == 0:
        writer = SummaryWriter(TENSORBOARD_LOG_DIR)
        # writer2 = SummaryWriter(TENSORBOARD_LOG_DIR + '/val-test-2w')
    
    if args.only_fc:
        model.eval()
    else:
        model.train()
    t0 = time.time()
    logger.info('rank %s, total_iter %s', hvd.rank(), total_iter)

    for epoch in range(all_epoches):
        for cur_iter, data in enumerate(loader):
            if args.fine_grained == 1:
                images, target, target2 = data
                target2 = target2.cuda(non_blocking=True)
            elif args.soft_label_gt == 1:
                images, target, target2 = data
                target2 = target2.cuda(non_blocking=True)
            else:
                images, target = data
                target = target.unsqueeze(1).repeat(1,4).reshape(-1)
                images = images.reshape(-1, 4, 224, 224)

            if isinstance(normalize_f, list):
                assert isinstance(images, list) and len(images) == len(normalize_f)
                images = [normf(img.cuda(non_blocking=True)) for img, normf in zip(images, normalize_f)]
            else:
                images = normalize_f(images.cuda(non_blocking=True))
            if isinstance(args.num_classes, list):
                target = [t.cuda(non_blocking=True) for t in target]
            else:
                target = target.cuda(non_blocking=True)

            if args.mixup > 0:
                images, target = utils.mixup(args.mixup, args.num_classes, images, target,
                                             one_hot=False if args.soft_label==1 or args.soft_label_gt==1 else True)
            target0 = target

            cur_epoch = total_iter / steps_per_epoch
            if cur_epoch > all_epoches:
                break

            lr = get_lr(base_lr, total_iter, lr_stages_step)
            if lr < 0:
                break
            set_lr(optimizer, lr)

            if isinstance(normalize_f, list):
                images = [img.contiguous() for img in images]
            else:
                images = images.contiguous()
            output = model(images)
            if args.fine_grained == 1:
                gw = get_w(args.fine_grained_w, total_iter, lr_stages_step)
                ww = 1 - (target2 == -1).float()
                target2 = target2.float() * ww
                loss0 = criterion(output[:, :args.num_classes], target)
                loss = loss0 + gw * sum([(ww[:, j] * cr(output[:, args.num_classes+j], target2[:, j])).mean()
                                   for j, cr in enumerate(criterion_fine)])
            elif args.soft_label_gt == 1:
                loss = (criterion(output, target) + criterion2(output, target2)) * 0.5
            else:
                if isinstance(args.num_classes, list):
                    loss = 0
                    j = 0
                    for c, t, n in zip(criterion, target, args.num_classes):
                        loss += c(output[:, j:j+n], t)
                        j += n
                else:
                    loss = criterion(output, target)
            if math.isnan(loss):
                raise RuntimeError("ERROR: Got NaN losses")
            if args.sam == 1:
                # first forward-backward pass
                optimizer.zero_grad()
                loss.backward()
                # optimizer.synchronize()
                optimizer_sam.first_step(zero_grad=True)
                # second forward-backward pass
                output2 = model(images)
                if args.fine_grained == 1:
                    loss02 = criterion(output2[:, :args.num_classes], target)
                    loss2 = loss02 + gw * sum([(ww[:, j] * cr(output2[:, args.num_classes+j], target2[:, j])).mean()
                                   for j, cr in enumerate(criterion_fine)])
                elif args.soft_label_gt == 1:
                    loss2 = (criterion(output2, target) + criterion2(output2, target2)) * 0.5
                else:
                    if isinstance(args.num_classes, list):
                        loss2 = 0
                        j = 0
                        for c, t, n in zip(criterion, target, args.num_classes):
                            loss2 += c(output2[:, j:j+n], t)
                            j += n
                    else:
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
                if isinstance(args.num_classes, list):
                    target0 = [torch.argmax(t, dim=-1) for t in target0]
                else:
                    target0 = torch.argmax(target0, dim=-1)
            elif args.soft_label_gt == 1:
                target0 = target2
            if args.fine_grained == 1:
                num_topks_correct = topks_correct(output[:, :args.num_classes], target0, [1])
            else:
                if isinstance(args.num_classes, list):
                    num_topks_correct = []
                    j = 0
                    for t, n in zip(target0, args.num_classes):
                        num_topks_correct.append(topks_correct(output[:, j:j+n], t, [1]))
                        j += n
                else:
                    num_topks_correct = topks_correct(output, target0, [1])
            if isinstance(args.num_classes, list):
                top1_acc_list = []
                for j in num_topks_correct:
                    top1_acc = j[0] / output.size(0)
                    if hvd.size() > 1:
                        loss = hvd.allreduce(loss)
                        top1_acc = hvd.allreduce(top1_acc)
                    top1_acc_list.append(top1_acc)
                top1_acc = top1_acc_list
            else:
                top1_acc = num_topks_correct[0] / output.size(0)
                if hvd.size() > 1:
                    loss = hvd.allreduce(loss)
                    top1_acc = hvd.allreduce(top1_acc)

            if hvd.rank() == 0:
                t = time.time()
                if isinstance(args.num_classes, list):
                    logger.info('epoch %.6f, iter %s, loss %.6f, top1_acc %.6f, lr %.6f, step time %.6f',
                                cur_epoch, total_iter, loss, top1_acc[0], lr, t-t0)
                    for j, v in enumerate(top1_acc):
                        writer.add_scalar('1-top1_acc%s' % j, v, total_iter)
                else:
                    logger.info('epoch %.6f, iter %s, loss %.6f, top1_acc %.6f, lr %.6f, step time %.6f',
                                cur_epoch, total_iter, loss, top1_acc, lr, t-t0)
                    writer.add_scalar('1-top1_acc', top1_acc, total_iter)
                if args.fine_grained == 1:
                    writer.add_scalar('2-loss0', loss0, total_iter)
                    writer.add_scalar('2-loss', loss, total_iter)
                    writer.add_scalar('2-gw', gw, total_iter)
                else:
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
        if lr < 0:
            break
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

