#!/usr/bin/env python3
import sys
import traceback
import os
import torch
import torchvision.transforms as transforms
import random
from urllib.request import urlopen
import io
from PIL import Image
from PIL import ImageFilter
from turbojpeg import TurboJPEG, TJPF_RGB
from transformation import resize_short, crop_image
from transformation import random_crop, distort_color, adjust_hue, blur, to_gray
import numpy as np
import cv2
from timm.data import create_transform
# from torchtoolbox.transform import Cutout

import hashlib

import horovod.torch as hvd
hvd.init()

try:
    _turbojpeg = TurboJPEG('/data1/liuyidi/moco/libturbojpeg.so.0.2.0')
except:
    _turbojpeg = TurboJPEG('/data/code/Moco/libturbojpeg.so.0.2.0')


def read_vid_cv2(path_to_vid):
    cap = cv2.VideoCapture(path_to_vid)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        break
    if len(frames) == 0:
        cap.release()
        return None
    return frames[0]

def map_file_code_to_seed(file_code):
    # 使用哈希函数（这里使用SHA256）将文件编码转换为整数
    hashed = hashlib.sha256(file_code.encode('utf-8')).hexdigest()
    seed = int(hashed, 16)  # 将哈希值转换为整数
    seed = seed % 429496729  #取模运算使得seed值在[0, 2^32-1]范围内

    return seed


class VideoSet(torch.utils.data.Dataset):
    
    def __init__(self, list_file, base_dir, transform=None, hvd=None, reader='PIL',
                 shuffle=True, list_repeat_n=1,frame_diff = False, line_num_mode=False, multi_label=False,
                 topk_soft=False, topk_soft_n=0, topk_soft_gt=False, fine_grained=False,
                 list_file_map=None, diff_mode=False, diff_mode_samp_2=None):
        '''
        list_file: str，列举所有训练数据的文本文件，每行为一个训练样本，通用格式为「文件名\t标签」，
                   文件名和标签的具体格式取决于其他参数，无特殊说明，文件名为单个图片（或者其他格式文件）
                   标签为整数（从0开始，标签类别序号）
        关于文件名: 多路输入时，每路文件之间用'&'分隔，
                  此基础上，每个部分，';'分隔的多个文件会被stack，即拓展出一维（若是2图片，则返回shape=2 h w 3），
                  此基础上，每个部分，可选包含'|'，其后内容解析为，
                      pos_[int](tmeta)，tmeta将传入transform作为位置信息参数，
                      npy-x_[int,int](xs,xe)，npy格式时指定最后输出sample[xs:xe]，
                      jpgs-[int]_[int,int](jpg_list_format_d,js,je)，见diff_mode的解释，
                  当list_file_map非None时，文件名中{[x]}将被替换，否则如果文件名开头为{[x]}将用于指定
                  使用base_dir中split(',')后的第x个元素
        关于标签: 依据line_num_mode、multi_label、topk_soft、topk_soft_gt、fine_grained的取值有不同格式，
                 见下面说明
        base_dir: str 或者 list（元素为str），数据文件夹地址，base_dir为list的情况见后面说明，为str时可为
                  ','分割的多个地址
        transform: 函数 或者 list（元素为函数），样本预处理函数，样本格式取决于其他参数，无特殊说明，为RGB图片，
                   此函数一般在训练时用于实现数据增强
        hvd: 传入hvd模块
        reader: str 或者 list（元素为str），取值范围['PIL', 'TurboJPEG', 'cv2', 'npy']，表示不同的数据读取方式
                取'PIL'时，读取结果为PIL图片，'TurboJPEG', 'cv2'时为图片numpy array（HWC），
                'npy'时为numpy array（shape取决于文件内容，可为任意值）
        关于多路文件: base_dir，transform，reader三者只能同时为或者非list，同时为list时，其len必须相同，
                    此时list_file中文件名包含'&'，且 .split('&') 后的len同base_dir的len，此时loader输出为
                    x, target, x为len>1的list
        shuffle: 是否对list_file内容进行随机打乱
        list_repeat_n: 重复读取list_file次数，对于数量特别少的list_file，使用此参数可以避免训练时频繁重启reader
        line_num_mode: 为 True 时，使用样本在list_file中的行号（从0开始）作为标签
        multi_label: 为 True 时，标签为'|'分隔的多个类标签
        topk_soft: 为 True 时（此时multi_label可为True），标签为软标签（每个类的概率值），topk_soft_n表示类别数，
                   此时标签的格式为（1000类为例）'123:0.6,9:0.3,877:0.08,32:0.02'，
                   表示第123/9/877/32类概率为0.6/0.3/0.08/0.02
        topk_soft_gt: 为 True 时，list_file每行格式 path_to_samp \t target \t target2，
                      其中target的解析同topk_soft为True时，表示软标签，target2为整数，表示硬标签
        fine_grained: 为 True 时，list_file每行格式 path_to_samp \t target \t target2，
                      其中target为整数，target2为多个整数并用','分隔，此格式用于标签粗分类和细粒度分类
        list_file_map: 非None时为str，指定一个多行文本文件，此时，list_file中的文件名中的{[xxx]}会被
                       替换为list_file_map中第xxx行的内容
        diff_mode: 为 True 时，diff_mode_samp_2为非None，list_file内容jpgs-[int]_[int,int](jpg_list_format_d,js,je)，
                   表示文件名为 xxx-%0[jpg_list_format_d]d.jpg(js到je)的一系列文件，该文件名数组输入diff_mode_samp_2
                   返回采样结果，此时的transform输入的x也为list，但transform输出为np array
        '''
        # parameters
        self.multi_input = 1
        if isinstance(base_dir, list) and isinstance(transform, list) and isinstance(reader, list):
            assert len(base_dir) > 1
            assert len(base_dir) == len(transform)
            assert len(base_dir) == len(reader)
            self.multi_input = len(base_dir)
            self.base_dir = [item.strip().split(',') for item in base_dir]
            for item in reader:
                assert item in ['PIL', 'TurboJPEG', 'cv2', 'npy']
        else:
            self.base_dir = base_dir.strip().split(',')
            assert reader in ['PIL', 'TurboJPEG', 'cv2', 'npy']
        self.transform = transform
        self.reader = reader
        if line_num_mode:
            self.label_mode = 'line_num_mode'
        else:
            self.label_mode = 'cls'
        if frame_diff:
            self.frame_diff = True
        # elif topk_soft:
        #     self.label_mode = 'topk_soft'
        #     self.topk_soft_n = topk_soft_n
        #     self.multi_label = multi_label
        # elif topk_soft_gt:
        #     self.label_mode = 'topk_soft_gt'
        #     self.topk_soft_n = topk_soft_n
        # elif fine_grained:
        #     self.label_mode = 'fine_grained'
        # else:
        #     self.label_mode = 'cls'
        #     self.multi_label = multi_label
        self.diff_mode = False

        # if diff_mode:
        #     self.diff_mode = True
        #     self.diff_mode_samp_2 = diff_mode_samp_2

        self.samples_str_map = []
        # if list_file_map is not None:
        #     with open(list_file_map, "r") as f:
        #         for l in f:
        #             self.samples_str_map.append(l.strip())
        #     print('read data from', list_file_map, len(self.samples_str_map))

        # read list file
        samples_str_i = [0]
        samples_line_n = []
        # # if not shuffle and hvd is not None and hvd.size() > 1:
        # #     self.samples_str = []
        # #     with open(list_file, "r") as f:
        # #         for i, l in enumerate(f):
        # #             if i % hvd.size() == hvd.rank():
        # #                 self.samples_str.append(l)
        # #                 samples_str_i.append(samples_str_i[-1]+len(l))
        # #                 samples_line_n.append(i)
        # #     self.samples_str = ''.join(self.samples_str)
        # else:
        with open(list_file, "r") as f:
            self.samples_str = f.read()
        with open(list_file, "r") as f:
            for i, l in enumerate(f):
                samples_str_i.append(samples_str_i[-1]+len(l))  #存储每行字符串的起止位置
                samples_line_n.append(i)
        assert len(self.samples_str) == samples_str_i[-1]
        self.samples_str_i = np.array(samples_str_i, dtype='int64')
        self.samples_line_n = np.array(samples_line_n, dtype='int32')
        samples_n = len(self.samples_line_n)
        del samples_str_i
        del samples_line_n
        print('read data from', list_file, samples_n)
        # samples' index
        self.samples_i = []
        for _ in range(list_repeat_n):
            samples_i = list(range(samples_n))
            if shuffle:
                random.shuffle(samples_i)
                if hvd is not None and hvd.size() > 1:
                    samples_i = hvd.broadcast_object(samples_i, root_rank=0)
                    samples_i = samples_i[
                        :len(samples_i)//hvd.size()*hvd.size()][
                         len(samples_i)//hvd.size()*hvd.rank()
                        :len(samples_i)//hvd.size()*(1+hvd.rank())]   ###给每个hvd.rank()分配数据子集
            else:
                if not self.label_mode == 'line_num_mode':
                    random.shuffle(samples_i)
                if hvd is not None and hvd.size() > 1:
                    samples_i = samples_i[:samples_n//hvd.size()]
            self.samples_i += samples_i
            del samples_i
        self.samples_i = np.array(self.samples_i, dtype='int32')
        print('hvd split', len(self.samples_i))

    def __getitem__(self, index):
        for i in range(20):
            index = self.samples_i[index]
            line = self.samples_str[self.samples_str_i[index]:self.samples_str_i[index+1]]
            line = line.strip()

            ls = line.split('\t')
            gt = ls[-1]
            data_list = ls[:-1]
        
            try:
                # info
                
                if self.label_mode == 'line_num_mode':
                    path_to_samp = line.split()[0]
                    target = self.samples_line_n[index]

                elif self.label_mode == 'cls':
                    target = int(gt)
                
                
                sample_list_list = []
                # path_to_samp_ori = path_to_samp.split('&')
                for mii in range(self.multi_input):
                    if self.multi_input == 1:
                        base_dir = self.base_dir
                        reader = self.reader
                        transform = self.transform
                    
                    sample_list = []
                    for path_to_samp in data_list:
                        
                        tmeta = None
                        xs = None
                        jpg_list_format_d = None
                        # xxxx.jpg|pos_[int](tmeta)
                        
                        # path_to_samp = path_to_samp[0]
                        # {[0-9]}xxxx.jpg

                        file_code = path_to_samp.split('/')[-1].split('-')[0]+path_to_samp.split('/')[-1].split('-')[1]+path_to_samp.split('/')[-1].split('-')[2]
                        seed_uid = map_file_code_to_seed(file_code)
            
                        
                        path_to_samp = base_dir[0] + path_to_samp
                        
                        
                        path_to_samp = [path_to_samp]
                        path_to_samp_list2 = path_to_samp
                        sample_list2 = []
                        for path_to_samp in path_to_samp_list2:
                            # read
                            
                           
                            f = open(path_to_samp, 'rb')
                            bs = f.read()
                            f.close()
                            # image decode
                            
                            if reader == 'TurboJPEG':
                                try:
                                    sample = _turbojpeg.decode(bs, pixel_format=TJPF_RGB)
                                except:
                                    sample = cv2.imdecode(np.frombuffer(bs, np.uint8), cv2.IMREAD_COLOR)
                                    if sample is None:
                                        sample = read_vid_cv2(path_to_samp) # for GIF
                                    sample = sample[:, :, ::-1] # BGR to RGB
                            
                           
                            # image decode /END
                            sample_list2.append(sample)
                        
                            sample = sample_list2[0]
                        if transform is not None:
                            try:
                                sample = transform(sample,seed = seed_uid)
                            except:
                                sample = transform(sample,seed = seed_uid)
                        sample_list.append(sample)
                    sample_list_list.append(sample_list)
            except Exception as e:
                print(traceback.format_exc())
                print(line, e)
                index = random.randint(0, len(self.samples_i)-1)
                continue
            
            # return
            sample_all = []
            for sample_list in sample_list_list:
                if len(sample_list) == 1:
                    sample = sample_list[0]
                else:
                    if self.frame_diff :
                       
                       # 计算帧差
                        frame_diffs = [cv2.absdiff(sample_list[i], sample_list[i+1]) for i in range(len(sample_list) - 1)] 

                       # 将帧差添加为第四个通道
                        images_with_diff = []

                       # 处理第一张图片
                        first_image_with_diff = np.zeros_like(sample_list[0][0])  # 创建与第一张图片相同尺寸的全零图像
                        first_image_with_diff = np.expand_dims(first_image_with_diff, axis=0)  
                        frame_diffs.append(first_image_with_diff)  # 将全零图像添加到帧差列表中

                        for i in range(len(frame_diffs)):
                            if i != 0:
                                # 计算灰度帧差

                                hwc_frame = frame_diffs[i-1].transpose(1, 2, 0)
                                gray_diff = cv2.cvtColor(hwc_frame, cv2.COLOR_RGB2GRAY)
                            
                                # 将灰度帧差归一化到[0, 1]区间
                                normalized_diff = gray_diff.astype(np.float32) / 255.0
                                normalized_diff = np.expand_dims(normalized_diff, axis=0)
                            else:
                                normalized_diff = frame_diffs[-1]
                            # 将归一化的帧差添加为第四个通道
                            image_with_diff = np.concatenate((sample_list[i-1], normalized_diff))
                            image_with_diff = torch.from_numpy(image_with_diff)
                            images_with_diff.append(image_with_diff)
                            image_with_diff = torch.stack(images_with_diff)

                        return image_with_diff, target

                        
                    else:
                        sample = torch.stack(sample_list)
                sample_all.append(sample)
            
            if len(sample_all) == 1:
                


                return sample_all[0], target
            
            


            return sample_all, target

            

    def __len__(self):
        return len(self.samples_i)


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def to_uint8(x):
    return (x * 255).type(torch.uint8)


def uint8_normalize_diff(x, dtype=torch.float,
                    mean=torch.tensor([0.485, 0.456, 0.406]).cuda(),
                    std=torch.tensor([0.229, 0.224, 0.225]).cuda(),
                    input_dim=3):

    if x.shape[-1] == input_dim:
        if len(x.shape) == 4:
            # NHWC -> NCHW
            x = x.permute(0,3,1,2)
        elif len(x.shape) == 5:
            # NTHWC -> NTCHW
            x = x.permute(0,1,4,2,3)
    if input_dim > 3:
        # assert input_dim % 3 == 0
        x[:,:3,:,:]= x[:,:3,:,:].type(dtype) / 255.0
        x[:,:3,:,:] -= mean.view(1, 3, 1, 1)
        x[:,:3,:,:] /= std.view(1 , 3, 1, 1)
        # x.cuda()
   
    
    return x











def val_cls_loader_frame_diff(list_file, base_dir, batch_size, threads, hvd, **args):
    size = 256
    if 'size' in args:
        size = args['size']
        del args['size']
    for k in list(args.keys()):
        if not args[k]:
            del args[k]

    
    def f(x, seed,tmeta=None):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # x = resize_short(x, 256)
        # x = crop_image(x, 224, center=True)
        #x = x[:x.shape[0]//2, :x.shape[1]//2, :] # c1
        #x = x[x.shape[0]//2:, :x.shape[1]//2, :] # c2
        #x = x[x.shape[0]//2:, x.shape[1]//2:, :] # c3
        #x = x[:x.shape[0]//2, x.shape[1]//2:, :] # c4
        x = cv2.resize(x, (size, size), interpolation = cv2.INTER_LINEAR)
        # return torch.from_numpy(x)
        x = x.transpose(2, 0, 1)
        return x
    dataset = VideoSet(list_file, base_dir, transform=f, hvd=hvd, reader='TurboJPEG',frame_diff=True, **args)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=threads,
        pin_memory=True,
        drop_last=False,)
    return dataset, loader


def train_cls_loader_frame_diff(list_file, base_dir, batch_size, threads, hvd, **args):
    size = 224
    if 'size' in args:
        size = args['size']
        del args['size']
    for k in list(args.keys()):
        if not args[k]:
            del args[k]
    randcorner = False
    if 'randcorner' in args:
        if args['randcorner'] == True:
            randcorner = True
        del args['randcorner']
    keep_wh_ratio = False
    if 'keep_wh_ratio' in args:
        if args['keep_wh_ratio'] == True:
            keep_wh_ratio = True
        del args['keep_wh_ratio']
    def f(x ,seed,tmeta=None,):
        

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


        if randcorner:
            tmeta = random.choice([0, 1,2,3,4])
        if tmeta is not None:
            x = random_crop(x, size=size, scale=(0.2, 1), flip_p=0.5, tmeta=tmeta, keep_wh_ratio=keep_wh_ratio)
        else:
            x = random_crop(x, size=size, scale=(0.2, 1), flip_p=0.5, keep_wh_ratio=keep_wh_ratio)
        if np.random.rand() < 0.5:
            shape0 = x.shape
            p0 = np.random.rand() * 0.8 + 0.2
            p1 = np.random.rand() * 0.8 + 0.2
            x = cv2.resize(x, (int(shape0[1]*p0), int(shape0[0]*p1)), interpolation=cv2.INTER_LINEAR)
            x = cv2.resize(x, (shape0[1], shape0[0]), interpolation=cv2.INTER_LINEAR)
        x = distort_color(x) # to CHW float32
        x = (np.clip(x, 0.0, 1.0) * 255).astype('uint8')
        # torch.manual_seed(torch.initial_seed())  ## reset seed
        return x
    
    dataset = VideoSet(list_file, base_dir, transform=f, hvd=hvd, reader='TurboJPEG',frame_diff=True, **args)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=threads,
        pin_memory=True,
        drop_last=True,)
    return dataset, loader


def train_cls_loader_4(list_file, base_dir, batch_size, threads, hvd, **args):
    size = 224
    if 'size' in args:
        size = args['size']
        del args['size']
    for k in list(args.keys()):
        if not args[k]:
            del args[k]
    randcorner = False
    if 'randcorner' in args:
        if args['randcorner'] == True:
            randcorner = True
        del args['randcorner']
    keep_wh_ratio = False
    if 'keep_wh_ratio' in args:
        if args['keep_wh_ratio'] == True:
            keep_wh_ratio = True
        del args['keep_wh_ratio']
    def f(x, tmeta=None):
        if randcorner:
            tmeta = random.choice([0, 1,2,3,4])
        if tmeta is not None:
            x = random_crop(x, size=size, scale=(0.2, 1), flip_p=0.5, tmeta=tmeta, keep_wh_ratio=keep_wh_ratio)
        else:
            x = random_crop(x, size=size, scale=(0.2, 1), flip_p=0.5, keep_wh_ratio=keep_wh_ratio)
        if np.random.rand() < 0.5:
            shape0 = x.shape
            p0 = np.random.rand() * 0.8 + 0.2
            p1 = np.random.rand() * 0.8 + 0.2
            x = cv2.resize(x, (int(shape0[1]*p0), int(shape0[0]*p1)), interpolation=cv2.INTER_LINEAR)
            x = cv2.resize(x, (shape0[1], shape0[0]), interpolation=cv2.INTER_LINEAR)
        x = distort_color(x) # to CHW float32
        x = (np.clip(x, 0.0, 1.0) * 255).astype('uint8')
        return torch.from_numpy(x)
    
    transform = create_transform(
            input_size= size,
            is_training=True,
            color_jitter= 0.4,
            auto_augment='rand-m9-mstd0.5-inc1' ,
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            interpolation='bicubic',
        )
    
    dataset = VideoSet(list_file, base_dir, transform=f, hvd=hvd, reader='TurboJPEG', **args)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=threads,
        pin_memory=True,
        drop_last=True,)
    return dataset, loader


def SpecAugment(x, F, mF, T, mT, p):
    if random.random() < p:
        for i in range(mF):
            f = random.randint(0, F)
            if f > x.shape[1]:
                f = x.shape[1]
            s = random.randint(0, x.shape[1]-f)
            x[:, s:s+f] = 0
    if random.random() < p:
        for i in range(mT):
            t = random.randint(0, T)
            if t > x.shape[0]:
                t = x.shape[0]
            s = random.randint(0, x.shape[0]-t)
            x[s:s+t, :] = 0













if __name__ == "__main__":
    # list_file = '/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/data*4_4_list.txt'
    # base_dir = '/data1/liuyidi/scene_cls/6b_2/'
    # # f = train_cls_loader_frame_diff
    # f= val_cls_loader_frame_diff

    # # dataset, loader = f(
    # #     list_file, base_dir,
    # #     batch_size=16, threads=32, hvd=hvd,
    # #     size=256, shuffle=False, diff_mode=0)
    

    # # f = val_cls_loader_4

    # dataset, loader = f(
    #     list_file, base_dir,
    #     batch_size=16, threads=32, hvd=hvd,
    #     size=224, shuffle=False,diff_mode=0,line_num_mode=True)
    # for cur_iter, (images, target) in enumerate(loader):
    #         images = images.contiguous()
    #         target = target.unsqueeze(1)
    #         targets = target.repeat(1,4)
    #         targets = targets.reshape(-1)


    #         images_2 = images.reshape(-1, 4, 224, 224)
    #         img = uint8_normalize_diff(images_2,input_dim=4)
    #         # images_ls = torch.unbind(images, dim=1)



    def f(x ,seed,tmeta=None,):
        
        
        
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)

        # tmeta = random.choice([0, 1,2,3,4])
        # x = random_crop(x, size=224, scale=(0.2, 1), flip_p=0.5, keep_wh_ratio=1)
        tmeta = random.choice([0, 1,2,3,4])
        if tmeta is not None:
            x = random_crop(x, size=224, scale=(0.2, 1), flip_p=0.5, tmeta=tmeta, keep_wh_ratio=1)
        
        if np.random.rand() < 0.5:
            shape0 = x.shape
            p0 = np.random.rand() * 0.8 + 0.2
            p1 = np.random.rand() * 0.8 + 0.2
            x = cv2.resize(x, (int(shape0[1]*p0), int(shape0[0]*p1)), interpolation=cv2.INTER_LINEAR)
            x = cv2.resize(x, (shape0[1], shape0[0]), interpolation=cv2.INTER_LINEAR)
        x = distort_color(x) # to CHW float32
        x = (np.clip(x, 0.0, 1.0) * 255).astype('uint8')
        # torch.manual_seed(torch.initial_seed())  ## reset seed
        return x
    img = cv2.imread('/data1/liuyidi/scene_cls/6b_2/20230507/20230507-2705764530-1354936142-0-1683413518-199760251-1683413517879.jpg')
    file_code = '20230507-2705764530-1354936142'
    seed = map_file_code_to_seed(file_code)
    img1 = f(img, seed=seed)
    img2 = f(img, seed=seed)
    result_consistent = np.array_equal(img1, img2)
    print("结果一致性：", result_consistent)

    s = 0
