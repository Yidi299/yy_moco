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

import torch.nn as nn
from loader import train_loader, uint8_normalize
from model import create_model, MoCo, BYOL, SimSiam
from precise_bn import get_bn_modules
from tensorboardX import SummaryWriter


_FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO, format=_FORMAT, stream=sys.stdout
)

logger = logging.getLogger(__name__)
logger.info('hvd info, size %s, rank %s, local_rank %s.', hvd.size(), hvd.rank(), hvd.local_rank())


def parse_args():
    parser = argparse.ArgumentParser()
    # options to ignore (auto added by training system)
    parser.add_argument("--data", type=str)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--hdfs-namenod", type=str)
    # Arguments
    parser.add_argument("--list-file", type=str, required=True)
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--net", type=str, default='resnest101')
    parser.add_argument("--alg", type=str, default='SimSiam')
    parser.add_argument("--pretrained-ckpt", type=str, default='')
    parser.add_argument("--base-lr", type=float, default=0.1)  # will be linear scaled by batch-size/256
    parser.add_argument("--lr-stages-step", type=str, default='6000,10000,13000,15000')
    parser.add_argument("--lr-warmup", type=float, default=0.0)
    parser.add_argument("--lr-warmup-step", type=int, default=0) # no warmup
    parser.add_argument("--ckpt-log-dir", type=str, default='ckpt-tmp')
    parser.add_argument("--ckpt-save-interval", type=int, default=1000)
    parser.add_argument("--fp16", type=int, default=0)
    parser.add_argument("--train-epoches", type=int, default=100)
    parser.add_argument("--no-bias-bn-wd", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    return parser.parse_args()


def topks_correct(preds, labels, list_of_top_k):
    ks = list_of_top_k
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [
        top_max_k_correct[:k, :].view(-1).float().sum() for k in ks
    ]
    return topks_correct


def load_last_checkpoint(dir_to_checkpoint, model, optimizer=None, name=None,two = False,another_op = None):
    if name is None:
        names = os.listdir(dir_to_checkpoint) if os.path.exists(dir_to_checkpoint) else []
        names = [f for f in names if "checkpoint" in f]
        if len(names) == 0:
            return None
        name = sorted(names)[-1]
    path_to_checkpoint = os.path.join(dir_to_checkpoint, name)
    # Load the checkpoint on CPU to avoid GPU mem spike.
    checkpoint = torch.load(path_to_checkpoint, map_location="cpu")
    if "model_state" not in checkpoint:
        # checkpoint = {"model_state": checkpoint}
        msg = model.load_state_dict(checkpoint['model'], strict=False)
    else:
        # model.load_state_dict(checkpoint["model_state"])
        msg = model.load_state_dict(checkpoint['model_state'], strict=False)
    if optimizer is not None:
        if two:
            optimizer_trans = optimizer
            optimizer_feature = another_op
            optimizer_trans.load_state_dict(checkpoint["optimizer_trans_state"])
            optimizer_feature.load_state_dict(checkpoint["optimizer_feature_state"])
        else:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "epoch" in checkpoint:
        epoch = checkpoint["epoch"]
        logger.info('checkpoint loaded from %s (epoch %.6f).', path_to_checkpoint, epoch)
        return epoch


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def get_lr(base_lr, cur_step, stages_step, lr_warmup=0, warmup_step=0):
    if cur_step < 0 or len(stages_step) == 0:
        return base_lr
    if cur_step >= stages_step[-1]:
        exit()
    if cur_step < warmup_step:
        ratio = cur_step / warmup_step
        return lr_warmup * (1 - ratio) + base_lr * ratio
    i = 0
    while cur_step > stages_step[i]:
        i += 1
    return base_lr * (0.1**i)


def get_loader(args):
    dataset, loader = train_loader(args.list_file, args.root_dir,
        batch_size=args.batch_size, threads=32, hvd=hvd, size=args.img_size)
    return loader


def main():
    args = parse_args()
    if hvd.rank() == 0:
        if not os.path.exists(args.ckpt_log_dir + '/log/'):
            os.makedirs(args.ckpt_log_dir + '/log/')
        f = open(args.ckpt_log_dir + '/log/para.txt.%s' % int(time.time()), 'w')
        print(args, file=f)
        f.close()
    base_lr = args.base_lr / 256 * args.batch_size * hvd.size()
    lr_stages_step = [int(i) for i in args.lr_stages_step.split(',') if i.strip() != '']
    TENSORBOARD_LOG_DIR = args.ckpt_log_dir + '/log'
    OUTPUT_DIR = args.ckpt_log_dir + '/ckpt'
    CHECKPOINT_PERIOD_STEPS = args.ckpt_save_interval
    all_epoches = args.train_epoches

    loader = get_loader(args)
    data_size = len(loader)
    logger.info('rank %s, data_size %s', hvd.rank(), data_size)
    steps_per_epoch = data_size

    if args.alg == 'SimSiam':
        alg_f = SimSiam
    elif args.alg == 'MoCo':
        alg_f = MoCo
    elif args.alg == 'BYOL':
        alg_f = BYOL
    model = create_model(arch=args.net, alg=alg_f, hvd=hvd,
                         pretrained_ckpt=args.pretrained_ckpt)

    if args.fp16 == 1:
        model = model.half().cuda()
        for bn in get_bn_modules(model):
            bn.float()
        normalize_f = lambda x: uint8_normalize(x, dtype=torch.half)
    else:
        model = model.cuda()
        normalize_f = uint8_normalize

    if args.no_bias_bn_wd == 1:
        bn_params = [v for n, v in model.named_parameters() if ('bn' in n or 'bias' in n)]
        rest_params = [v for n, v in model.named_parameters() if not ('bn' in n or 'bias' in n)]
        optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0 },
                                     {'params': rest_params, 'weight_decay': args.weight_decay}],
                                    base_lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    base_lr, momentum=0.9, weight_decay=args.weight_decay)
    if hvd.size() > 1:
        # Add Horovod Distributed Optimizer
        optimizer = hvd.DistributedOptimizer(optimizer,
                        named_parameters=model.named_parameters())

    total_iter = 0
    if hvd.rank() == 0:
        epoch = load_last_checkpoint(OUTPUT_DIR, model, optimizer)
        if epoch:
            total_iter = int(epoch * steps_per_epoch)
    if hvd.size() > 1:
        # Broadcast parameters from rank 0 to all other processes.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        total_iter = hvd.broadcast_object(total_iter, root_rank=0)
    ckp_save_iter = total_iter

    if hvd.rank() == 0:
        writer = SummaryWriter(TENSORBOARD_LOG_DIR)

    model.train()
    t0 = time.time()
    logger.info('rank %s, total_iter %s', hvd.rank(), total_iter)

    for epoch in range(all_epoches):
        for cur_iter, (images, _) in enumerate(loader):

            cur_epoch = total_iter / steps_per_epoch
            if cur_epoch > all_epoches:
                break

            lr = get_lr(base_lr, total_iter, lr_stages_step, args.lr_warmup, args.lr_warmup_step)
            set_lr(optimizer, lr)

            loss = model(x1=images[0], x2=images[1], norm_f=normalize_f)

            if math.isnan(loss):
                raise RuntimeError("ERROR: Got NaN losses")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if hvd.size() > 1:
                loss = hvd.allreduce(loss)

            if hvd.rank() == 0:
                t = time.time()
                logger.info('epoch %.6f, iter %s, loss %.6f, lr %.6f, step time %.6f',
                            cur_epoch, total_iter, loss, lr, t-t0)
                writer.add_scalar('1-loss', loss, total_iter)
                writer.add_scalar('2-lr', lr, total_iter)
                writer.add_scalar('3-steps_per_s', 1.0 / (t-t0), total_iter)
                t0 = t

            if hvd.rank() == 0 and (total_iter == 0
                or total_iter - ckp_save_iter >= CHECKPOINT_PERIOD_STEPS):
                ckp_save_iter = total_iter
                path_to_checkpoint = OUTPUT_DIR + '/' + 'checkpoint-iter-%06d.pyth' % total_iter
                if not os.path.exists(OUTPUT_DIR):
                    os.makedirs(OUTPUT_DIR)
                checkpoint = {
                    "epoch": cur_epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "hvd.size": hvd.size(),}
                torch.save(checkpoint, path_to_checkpoint)
                logger.info('save checkpoint at step %s.', total_iter)

            total_iter += 1
        loader = get_loader(args) # shuffle data

    if hvd.rank() == 0:
        ckp_save_iter = total_iter
        path_to_checkpoint = OUTPUT_DIR + '/' + 'checkpoint-iter-%06d.pyth' % total_iter
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        checkpoint = {
            "epoch": cur_epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "hvd.size": hvd.size(),}
        torch.save(checkpoint, path_to_checkpoint)
        logger.info('save checkpoint at step %s.', total_iter)


if __name__ == "__main__":
    main()

