#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.models as models
from loader import uint8_normalize
import logging

import timm
logger = logging.getLogger(__name__)


def convert_sync_batchnorm(module, hvd):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = hvd.SyncBatchNorm(module.num_features,
                                          module.eps, module.momentum,
                                          module.affine,
                                          module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, convert_sync_batchnorm(child, hvd))
    del module
    return module_output


# utils
@torch.no_grad()
def concat_all_gather(tensor, hvd):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    return hvd.allgather(tensor)


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.2, mlp=True,
                 hvd=None, pretrained_ckpt=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.2)
        """
        assert hvd is not None and hvd.size() > 1, 'hvd is None or hvd.size is 1'

        super(MoCo, self).__init__()

        self.hvd = hvd
        self.K = K
        self.m0 = m
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        
        if pretrained_ckpt is not None:
            checkpoint = torch.load(pretrained_ckpt, map_location="cpu")
            if "fc.weight" in checkpoint:
                del checkpoint["fc.weight"]
                del checkpoint["fc.bias"]
            msg = self.encoder_q.load_state_dict(checkpoint, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, set(msg.missing_keys)
            logger.info('load_finetune_checkpoint from %s.', pretrained_ckpt)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys, hvd=self.hvd)

        batch_size = keys.shape[0]
        self.m = self.m0 ** (batch_size / 256)

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x, hvd=self.hvd)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all) # on CPU

        # broadcast to all gpus
        idx_shuffle = self.hvd.broadcast_object(idx_shuffle, root_rank=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = self.hvd.rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x, hvd=self.hvd)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = self.hvd.rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, norm_f=uint8_normalize):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(norm_f(im_q))  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            im_k = norm_f(im_k.cuda(non_blocking=True))
            idx_unshuffle = idx_unshuffle.cuda(non_blocking=True)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

class BYOL(nn.Module):
    """
    https://arxiv.org/abs/2006.07733
    """
    def __init__(self, base_encoder, dim=256, m=0.999, shuffle_bn=False, hvd=None, pretrained_ckpt=None):
        """
        dim: feature dimension (default: 256)
        m: moving avg of updating target encoder (default: 0.999)
        """
        super(BYOL, self).__init__()

        self.hvd = hvd
        self.m0 = m
        self.m = self.m0
        self.shuffle_bn = shuffle_bn

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_online = base_encoder(num_classes=dim)
        self.encoder_target = base_encoder(num_classes=dim)

        if pretrained_ckpt is not None:
            checkpoint = torch.load(pretrained_ckpt, map_location="cpu")
            if "fc.weight" in checkpoint:
                del checkpoint["fc.weight"]
                del checkpoint["fc.bias"]
            msg = self.encoder_online.load_state_dict(checkpoint, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, set(msg.missing_keys)
            logger.info('load_finetune_checkpoint from %s.', pretrained_ckpt)

        # hack: brute-force replacement
        dim_mlp1 = self.encoder_online.fc.weight.shape[1] # 2048
        dim_mlp2 = 4096
        self.encoder_online.fc = nn.Sequential(
            nn.Linear(dim_mlp1, dim_mlp2), nn.BatchNorm1d(dim_mlp2), nn.ReLU(), nn.Linear(dim_mlp2, dim))
        self.encoder_online_q  = nn.Sequential(
            nn.Linear(dim,      dim_mlp2), nn.BatchNorm1d(dim_mlp2), nn.ReLU(), nn.Linear(dim_mlp2, dim))
        self.encoder_target.fc = nn.Sequential(
            nn.Linear(dim_mlp1, dim_mlp2), nn.BatchNorm1d(dim_mlp2), nn.ReLU(), nn.Linear(dim_mlp2, dim))

        for param_q, param_k in zip(self.encoder_online.parameters(),
                                    self.encoder_target.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_online.parameters(),
                                    self.encoder_target.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x, hvd=self.hvd)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all) # on CPU

        # broadcast to all gpus
        idx_shuffle = self.hvd.broadcast_object(idx_shuffle, root_rank=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = self.hvd.rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x, hvd=self.hvd)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = self.hvd.rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    # https://github.com/Spijkervet/BYOL
    def _loss_fn(self, x, y):
        x = nn.functional.normalize(x, dim=-1, p=2)
        y = nn.functional.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, im_q, im_k, norm_f=uint8_normalize):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        self.m = self.m0 ** (im_q.shape[0]*self.hvd.size()/512)
        nim_q = norm_f(im_q.cuda(non_blocking=True))
        nim_k = norm_f(im_k.cuda(non_blocking=True))
        # compute query features
        q1 = self.encoder_online_q(self.encoder_online(nim_q)) # queries: NxC
        q2 = self.encoder_online_q(self.encoder_online(nim_k))

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()      # update the key encoder
            if self.shuffle_bn:
                im_k, idx_unshuffle_k = self._batch_shuffle_ddp(im_k)
                im_q, idx_unshuffle_q = self._batch_shuffle_ddp(im_q)
                im_k = norm_f(im_k.cuda(non_blocking=True))
                im_q = norm_f(im_q.cuda(non_blocking=True))
                k1 = self.encoder_target(im_k).detach()  # keys: NxC
                k2 = self.encoder_target(im_q).detach()
                k1 = self._batch_unshuffle_ddp(k1, idx_unshuffle_k)
                k2 = self._batch_unshuffle_ddp(k2, idx_unshuffle_q)
            else:
                k1 = self.encoder_target(nim_k).detach()  # keys: NxC
                k2 = self.encoder_target(nim_q).detach()

        loss = self._loss_fn(q1, k1) + self._loss_fn(q2, k2)
        return loss.mean()


class SimSiam(nn.Module):
    """
    http://arxiv.org/abs/2011.10566
    """
    def __init__(self, base_encoder, hvd=None, pretrained_ckpt=None):
        super(SimSiam, self).__init__()

        self.hvd = hvd

        # create the encoders
        self.encoder = base_encoder

        if pretrained_ckpt is not None and pretrained_ckpt != '':
            checkpoint = torch.load(pretrained_ckpt, map_location="cpu")
            if "fc.weight" in checkpoint:
                del checkpoint["fc.weight"]
                del checkpoint["fc.bias"]
            msg = self.encoder.load_state_dict(checkpoint, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, set(msg.missing_keys)
            logger.info('load_finetune_checkpoint from %s.', pretrained_ckpt)

        # hack: brute-force replacement
        dim_in = self.encoder.head.weight.shape[1] # 2048
        dim = 2048
        self.encoder.head = nn.Sequential(
            nn.Linear(dim_in, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(   dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(   dim, dim), nn.BatchNorm1d(dim))
        dim_pred = 512
        self.predictor  = nn.Sequential(
            nn.Linear(dim, dim_pred), nn.BatchNorm1d(dim_pred), nn.ReLU(),
            nn.Linear(dim_pred, dim))

    def _loss_f(self, p, z):
        z = z.detach() # stop gradient
        p = nn.functional.normalize(p, dim=1, p=2)
        z = nn.functional.normalize(z, dim=1, p=2)
        return -(p * z).sum(dim=1).mean()

    def forward(self, x1, x2, norm_f=uint8_normalize):
        x1 = norm_f(x1.cuda(non_blocking=True))
        x2 = norm_f(x2.cuda(non_blocking=True))

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss = (self._loss_f(p1, z2) + self._loss_f(p2, z1)) / 2
        return loss


def create_model(arch='resnet50', alg=MoCo, hvd=None, **keyargs):
    print(arch+"1111111111")
    if arch == 'resnet50_x2':
        from resnet_x2 import resnet50
        net = resnet50
    elif arch.startswith('resnest'):
        import resnest
        net = resnest.__dict__[arch]
    elif arch == 'swinv2_small_window8_256':
        net = timm.create_model(arch,num_classes=1,input_size=(3, 192, 192))
        
        
    else:
        net = models.__dict__[arch]
    model = alg(net, hvd=hvd, **keyargs)
    return model


def create_model2(arch='resnet50', alg=BYOL, hvd=None, **keyargs):
    return create_model(arch, alg, hvd, **keyargs)


def create_model3(arch='resnet50', alg=SimSiam, hvd=None, **keyargs):
    return create_model(arch, alg, hvd, **keyargs)

