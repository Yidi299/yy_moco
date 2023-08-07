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

from transformer_block import ImageTransformer
from Lstm_block import TimeSeriesClassifier

from loader import train_cls_loader, val_cls_loader, uint8_normalize
from loader import train_diff_cls_loader, val_diff_cls_loader
from loader import train_cls_loader_npy, val_cls_loader_npy
from loader import train_cls_loader_npy_img, val_cls_loader_npy_img
from loader_my import train_cls_loader_4, val_cls_loader_4
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
    parser.add_argument("--list-file", type=str, default='/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/data*4_4_list.txt')
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
    parser.add_argument("--pretrained-ckpt", type=str, default='/data1/liuyidi/scene_cls/V4.1/log_dir/V4.1_test27_1_fix/ckpt/checkpoint-iter-008000.pyth')
    parser.add_argument("--only-fc", type=int, default=0) # optimize only the linear classifier
    parser.add_argument("--base-lr", type=float, default=0.01)  # will be linear scaled by batch-size/256
    parser.add_argument("--lr-stages-step", type=str, default='4000,6000,8001')
    parser.add_argument("--ckpt-log-dir", type=str, default='/data1/liuyidi/scene_cls/V4.1/log_dir/V4.2_duibi2')
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
    parser.add_argument("--sam", type=int, default=0)
    parser.add_argument("--rand-corner", type=int, default=1)
    parser.add_argument("--loader-keep-wh-ratio", type=int, default=0)
    parser.add_argument("--diff-mode", type=int, default=0)
    
    return parser.parse_args()




class CustomModel(nn.Module):
    def __init__(self, pretrained_model, num_classes,input_features, dropout_rate=0.5):
        super(CustomModel, self).__init__()
        
        self.pretrained_model = pretrained_model
        self.pretrained_model.eval()

        self.linear_layer = nn.Linear(input_features * 4, input_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_layer2 = nn.Linear(input_features , num_classes)
        
    def forward(self, imgs):
        # 假设imgs的形状为 (batch_size, 4, C, H, W)
        batch_size, _, C, H, W = imgs.shape
        imgs = imgs.view(-1, C, H, W)  # 将图片展开为 (batch_size * 4, C, H, W)
        
        with torch.no_grad():
            features = self.pretrained_model(imgs)  # 提取特征 (batch_size * 4, output_dim)
        
        features_grouped = features.view(batch_size, -1)  # 将特征按组分组 (batch_size, output_dim * 4)
        
        features_grouped = self.linear_layer(features_grouped)  # 输出 (batch_size, output_dim)

        features_grouped = self.dropout(features_grouped)  # 添加 dropout 

        out = self.linear_layer2(features_grouped)  # 输出 (batch_size, num_classes)
        
        return out
    

class CustomModel2(nn.Module):
    def __init__(self, pretrained_model, num_classes,input_features, dropout_rate=0.5):
        super(CustomModel2, self).__init__()
        
        self.pretrained_model = pretrained_model
        self.pretrained_model.eval()

        
        self.con_1d_layer = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)
        # self.dropout = nn.Dropout(dropout_rate)
        self.linear_layer2 = nn.Linear(input_features , num_classes)
        
    def forward(self, imgs):
        print(imgs.shape)
        # 假设imgs的形状为 (batch_size, 4, C, H, W)
        batch_size, _, C, H, W = imgs.shape
        imgs0 = imgs.view(-1, C, H, W)  # 将图片展开为 (batch_size * 4, C, H, W)
        
        with torch.no_grad():
            features = self.pretrained_model(imgs0)  # 提取特征 (batch_size * 4, output_dim)
        
        features_grouped = features.view(batch_size,4, -1)  # 将特征按组分组 (batch_size, output_dim * 4)
        
        features_grouped = self.con_1d_layer(features_grouped)  # 输出 (batch_size, output_dim)

        features_grouped = features_grouped.view(batch_size, -1)  

        # features_grouped = self.dropout(features_grouped)  # 添加 dropout 

        out = self.linear_layer2(features_grouped)  # 输出 (batch_size, num_classes) 
        s= 0
        
        return out


class CustomModel3(nn.Module):
    def __init__(self, pretrained_model, num_classes,input_features, dropout_rate=0.5):
        super(CustomModel3, self).__init__()
        
        self.pretrained_model = pretrained_model
        self.pretrained_model.eval()

        
        # self.con_1d_layer = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)
        # self.dropout = nn.Dropout(dropout_rate)
        self.linear_layer2 = nn.Linear(input_features , num_classes)
        
    def forward(self, imgs):
        print(imgs.shape)
        # 假设imgs的形状为 (batch_size, 4, C, H, W)
        batch_size, _, C, H, W = imgs.shape
        imgs0 = imgs.view(-1, C, H, W)  # 将图片展开为 (batch_size * 4, C, H, W)
        
        with torch.no_grad():
            features = self.pretrained_model(imgs0)  # 提取特征 (batch_size * 4, output_dim)
        
        features_grouped = features.view(batch_size,4, -1)  # 将特征按组分组 (batch_size, output_dim * 4)
        
        # features_grouped = self.con_1d_layer(features_grouped)  # 输出 (batch_size, output_dim)

        # features_grouped = features_grouped.view(batch_size, -1)  

        # features_grouped = self.dropout(features_grouped)  # 添加 dropout 

        # out = self.linear_layer2(features_grouped)  # 输出 (batch_size, num_classes)


        # images_ls = torch.unbind(imgs, dim=1)
        # imgs_2 = images_ls[-1]

        # imgs_2 = imgs[:,3,:,:,:]
        # with torch.no_grad():
        #     out = self.pretrained_model(imgs_2) 

        fea0 = features_grouped[:,0,:]
        out =  self.linear_layer2(fea0)
        
        return out


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


def get_loader(args, threads_gpu=8):
    
    f = train_cls_loader_4
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
    # if len(args.num_classes) == 1:
    #     args.num_classes = args.num_classes[0]
    args.num_classes = 43
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
        model0 = net_f(num_classes=43, **kwargs).cuda()
        
    #加载model0去掉fc层的预训练参数
    if hvd.rank() == 0:
        load_finetune_checkpoint(PRETRAINED_PATH, model0, remove_fc=True, input_dim=3)
    
        
        

    model0 = nn.Sequential(*list(model0.children())[:-1])

    # model = CustomModel2(model0, args.num_classes,input_features=2048).cuda()
    model = CustomModel3(model0,args.num_classes,input_features=2048).cuda()
    
    if hvd.rank() == 0:   
    # #修改权重
        cls_weight = torch.load('/data1/liuyidi/scene_cls/V4.1/log_dir/V4.1_test27_1_fix/ckpt/checkpoint-iter-008000.pyth', map_location="cpu")
        # #将cls_weight最后一层的参数赋值给model的最后一层
        model.linear_layer2.weight.data = cls_weight['model_state']['fc.weight'].data.cuda()
        model.linear_layer2.bias.data = cls_weight['model_state']['fc.bias'].data.cuda()
    
    for name, param in model.named_parameters():
        if (name.startswith('pretrained_model')):
            param.requires_grad = False
        # if (name.startswith('feature_extractor')):
        #     param.requires_grad = False
        # elif name =='con_1d_layer.weight':
        #     param.requires_grad = False
        #     param[0, 0:4, 0] = torch.tensor([0.0, 0.0, 0.0, 1.0])

        # elif name =='con_1d_layer.bias':
        #     param.requires_grad = False
        #     param[0] = torch.tensor([0.0])
               
    
    
    
    model = model.float().cuda()
       
    normalize_f = lambda x: uint8_normalize(x, input_dim=3 + 3 * args.diff_mode)
    
    
    criterion = nn.CrossEntropyLoss().cuda()
    
    if args.sam == 1:
        optim_f = lambda params, **kwargs: SAM(params, torch.optim.SGD, rho=0.05, **kwargs)
    else:
        optim_f = torch.optim.SGD

    
    parameters = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
    # assert len(parameters) >= 4
    # optimizer = optim_f(parameters, lr=base_lr, momentum=0.9, weight_decay=0.0)
    
    optimizer = optim_f(model.parameters(),
                                lr=base_lr, momentum=0.9, weight_decay=args.weight_decay)
    
    for parm in model.parameters():
        print(parm.requires_grad)

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
            
            
        #     load_finetune_checkpoint(PRETRAINED_PATH, model, remove_fc=True, input_dim=3)
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

    for epoch in range(all_epoches):
        for cur_iter,  (images_ls, target) in enumerate(loader):
            target = target.cuda(non_blocking=True)
            target0 = target
            cur_epoch = total_iter / steps_per_epoch
            if cur_epoch > all_epoches:
                break

            lr = get_lr(base_lr, total_iter, lr_stages_step)
            if lr < 0:
                break
            set_lr(optimizer, lr)

            images = images_ls.contiguous()
            images = images.cuda(non_blocking=True).to(torch.float)
            output = model(images)
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
    


