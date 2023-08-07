#!/usr/bin/env python3
import torch
import torch.nn as nn
from resnest.resnet import Bottleneck
import logging
logger = logging.getLogger(__name__)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)



class CnnHead(nn.Module):
    def __init__(self, arch, num_classes, conv_blocks=1, final_drop=0.0, dropblock_prob=0, reduce_dim=None):
        super(CnnHead, self).__init__()
        self.arch = arch(num_classes = 0)
        self.conv = nn.Sequential(*[
                        Bottleneck(self.arch.inplanes, 512,
                                radix=self.arch.radix, cardinality=self.arch.cardinality,
                                bottleneck_width=self.arch.bottleneck_width,
                                avd=self.arch.avd, avd_first=self.arch.avd_first,
                                dilation=1, rectified_conv=self.arch.rectified_conv,
                                rectify_avg=self.arch.rectify_avg,
                                norm_layer=nn.BatchNorm2d, dropblock_prob=dropblock_prob,
                                last_gamma=self.arch.last_gamma)
                        for i in range(conv_blocks)])
        self.avgpool = GlobalAvgPool2d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        if reduce_dim is not None:
            self.fc2 = nn.Linear(512 * Bottleneck.expansion, reduce_dim)
            if num_classes > 0:
                self.fc = nn.Linear(reduce_dim, num_classes)
            elif num_classes == 0:
                self.fc = nn.Identity()
            else:
                self.fc = -1
        else:
            self.fc2 = None
            if num_classes > 0:
                self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
            elif num_classes == 0:
                self.fc = nn.Identity()
            else:
                self.fc = -1

    def forward(self, xs):
        # xs: N T C H W
        feamaps = []
        for i in range(xs.shape[1]):
            feamaps.append(self.arch(xs[:,i,:,:,:]))
        conv_in = torch.stack(feamaps, dim=-1)
        conv_in = torch.unsqueeze(conv_in, dim=-1)
        x = self.conv(conv_in)
        #x = conv_in
        
        if self.fc != -1:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if self.fc2 is not None:
                x = self.fc2(x)
            if self.drop:
                x = self.drop(x)
            x = self.fc(x)
        
        return x


class CnnHead2(nn.Module):
    def __init__(self, arch, num_classes, conv_blocks=1, final_drop=0.0, dropblock_prob=0, reduce_dim=None):
        super(CnnHead2, self).__init__()
        self.arch = arch(num_classes = 0)
        self.arch2 = arch(num_classes = 0)
        self.fc0 = nn.Linear(512 * Bottleneck.expansion * 2, 512 * Bottleneck.expansion)
        self.conv = nn.Sequential(*[
                        Bottleneck(self.arch.inplanes, 512,
                                radix=self.arch.radix, cardinality=self.arch.cardinality,
                                bottleneck_width=self.arch.bottleneck_width,
                                avd=self.arch.avd, avd_first=self.arch.avd_first,
                                dilation=1, rectified_conv=self.arch.rectified_conv,
                                rectify_avg=self.arch.rectify_avg,
                                norm_layer=nn.BatchNorm2d, dropblock_prob=dropblock_prob,
                                last_gamma=self.arch.last_gamma)
                        for i in range(conv_blocks)])
        self.avgpool = GlobalAvgPool2d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        if reduce_dim is not None:
            self.fc2 = nn.Linear(512 * Bottleneck.expansion, reduce_dim)
            if num_classes > 0:
                self.fc = nn.Linear(reduce_dim, num_classes)
            elif num_classes == 0:
                self.fc = nn.Identity()
            else:
                self.fc = -1
        else:
            self.fc2 = None
            if num_classes > 0:
                self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
            elif num_classes == 0:
                self.fc = nn.Identity()
            else:
                self.fc = -1

    def forward(self, xs):
        xs, xs2 = xs
        assert xs.shape[0] == xs2.shape[0]
        assert xs.shape[1] == xs2.shape[1]
        # xs: N T C H W
        feamaps = []
        for i in range(xs.shape[1]):
            fea1 = self.arch(xs[:,i,:,:,:])
            fea2 = self.arch2(xs2[:,i,:,:,:])
            fea = torch.cat([fea1, fea2], dim=1)
            feamaps.append(self.fc0(fea))
        conv_in = torch.stack(feamaps, dim=-1)
        conv_in = torch.unsqueeze(conv_in, dim=-1)
        x = self.conv(conv_in)
        #x = conv_in

        if self.fc != -1:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if self.fc2 is not None:
                x = self.fc2(x)
            if self.drop:
                x = self.drop(x)
            x = self.fc(x)

        return x
