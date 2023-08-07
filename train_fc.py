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
from loader import train_cls_loader, uint8_normalize
from precise_bn import update_bn_stats
from tensorboardX import SummaryWriter


_FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO, format=_FORMAT, stream=sys.stdout
)

logger = logging.getLogger(__name__)
logger.info('hvd info, size %s, rank %s, local_rank %s.', hvd.size(), hvd.rank(), hvd.local_rank())


from train_self_superv import parse_args, topks_correct, set_lr, get_lr, load_last_checkpoint


def load_finetune_checkpoint(dir_to_checkpoint, model):
    # NAMR_PRE = 'encoder_q.'
    NAMR_PRE = 'encoder_online.'
    names = os.listdir(dir_to_checkpoint) if os.path.exists(dir_to_checkpoint) else []
    names = [f for f in names if "checkpoint" in f]
    if len(names) == 0:
        return None
    name = sorted(names)[-1]
    path_to_checkpoint = os.path.join(dir_to_checkpoint, name)
    # Load the checkpoint on CPU to avoid GPU mem spike.
    checkpoint = torch.load(path_to_checkpoint, map_location="cpu")
    pretrained_dict = {k.replace(NAMR_PRE, ''): v
                        for k, v in checkpoint['model_state'].items()
                        if k.replace(NAMR_PRE, '') in model.state_dict()} ## encoder_q or module.encoder_q
    msg = model.load_state_dict(pretrained_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, set(msg.missing_keys)
    epoch = checkpoint["epoch"]
    logger.info('load_finetune_checkpoint from %s (epoch %.6f).', path_to_checkpoint, epoch)
    return epoch


def calculate_and_update_precise_bn(model, loader, num_iters=100):
    def _gen_loader():
        while True:
            for cur_iter, (images, target) in enumerate(loader):
                images = uint8_normalize(images.cuda(non_blocking=True))
                yield images
    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), hvd=hvd, num_iters=num_iters)


def get_loader(batch_size):
    dataset, loader = train_cls_loader(
        'data/train.txt',
        'http://filer.ai.yy.com:9889/dataset/heliangliang/imagenet/train/',
        batch_size=batch_size, threads=32, hvd=hvd)
    return loader


def main():
    args = parse_args()
    batch_size = 64
    base_lr = 30 / 256 * batch_size * hvd.size()
    TENSORBOARD_LOG_DIR = './ckpt-byol/imagenet-lr-0.1-shufflebn/log-fc'
    OUTPUT_DIR          = './ckpt-byol/imagenet-lr-0.1-shufflebn/ckpt-fc'
    PRETRAINED_DIR      = './ckpt-byol/imagenet-lr-0.1-shufflebn/ckpt'
    CHECKPOINT_PERIOD_STEPS = 1000
    all_epochs = 50

    loader = get_loader(batch_size)
    data_size = len(loader)
    logger.info('rank %s, data_size %s', hvd.rank(), data_size)
    steps_per_epoch = data_size

    # create model
    import torchvision.models as models
    model = models.__dict__['resnet50']().cuda()
    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    criterion = nn.CrossEntropyLoss().cuda()

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, base_lr, momentum=0.9, weight_decay=0.0)

    if hvd.size() > 1:
        # Add Horovod Distributed Optimizer
        optimizer = hvd.DistributedOptimizer(optimizer,
                        named_parameters=[item for item in model.named_parameters()
                                          if item[1].requires_grad])

    total_iter = 0
    epoch = load_last_checkpoint(OUTPUT_DIR, model, optimizer)
    if epoch:
        total_iter = int(epoch * steps_per_epoch)
    else:
        load_finetune_checkpoint(PRETRAINED_DIR, model)
        model.train()
        calculate_and_update_precise_bn(model, loader)
    ckp_save_iter = total_iter

    if hvd.rank() == 0:
        writer = SummaryWriter(TENSORBOARD_LOG_DIR)

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()
    t0 = time.time()
    logger.info('rank %s, total_iter %s', hvd.rank(), total_iter)

    for epoch in range(all_epochs):
        for cur_iter, (images, target) in enumerate(loader):
            images = uint8_normalize(images.cuda(non_blocking=True))
            target = target.cuda(non_blocking=True)

            cur_epoch = total_iter / steps_per_epoch
            if cur_epoch > all_epochs:
                break

            lr = get_lr(cur_epoch, base_lr, all_epochs)
            set_lr(optimizer, lr)

            output = model(images)
            loss = criterion(output, target)
            if math.isnan(loss):
                raise RuntimeError("ERROR: Got NaN losses")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_topks_correct = topks_correct(output, target, [1])
            top1_acc = num_topks_correct[0] / output.size(0)
            if hvd.size() > 1:
                loss = hvd.allreduce(loss)
                top1_acc = hvd.allreduce(top1_acc)

            if hvd.rank() == 0:
                t = time.time()
                logger.info('epoch %.6f, iter %s, loss %.6f, top1_acc %.6f, lr %.6f, step time %.6f',
                            cur_epoch, total_iter, loss, top1_acc, lr, t-t0)
                writer.add_scalar('1-top1_acc', top1_acc, total_iter)
                writer.add_scalar('2-loss', loss, total_iter)
                writer.add_scalar('3-lr', lr, total_iter)
                writer.add_scalar('4-steps_per_s', 1.0 / (t-t0), total_iter)
                t0 = t
            if total_iter == 0 or total_iter - ckp_save_iter >= CHECKPOINT_PERIOD_STEPS:
                if hvd.rank() == 0:
                    if not os.path.exists(OUTPUT_DIR):
                        os.makedirs(OUTPUT_DIR)
                    checkpoint = {
                        "epoch": cur_epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "hvd.size": hvd.size(),}
                    path_to_checkpoint = OUTPUT_DIR + '/' + 'checkpoint-iter-%06d.pyth' % total_iter
                    torch.save(checkpoint, path_to_checkpoint)
                    logger.info('save checkpoint at step %s.', total_iter)
                ckp_save_iter = total_iter
            total_iter += 1
        loader = get_loader(batch_size) # shuffle data


if __name__ == "__main__":
    main()

