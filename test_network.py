#!/usr/bin/env python3
import torch
import horovod.torch as hvd

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
#torch.cuda.set_device(hvd.local_rank())

import argparse
import sys
import torch
import logging
import time
import math
import random


_FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO, format=_FORMAT, stream=sys.stdout
)

logger = logging.getLogger(__name__)
logger.info('hvd info, size %s, rank %s, local_rank %s.', hvd.size(), hvd.rank(), hvd.local_rank())


def loader_download(list_file, base_dir, batch_size, threads):
    
    import io
    from urllib.request import urlopen
    
    class VideoSet(torch.utils.data.Dataset):
        def __init__(self, list_file, base_dir):
            with open(list_file, "r") as f:
                self._path_to_samples = [line.split()[0] for line in f]
            random.shuffle(self._path_to_samples)
            self._base_dir = base_dir
        
        def __getitem__(self, index):
            try:
                path_to_samp = self._base_dir + self._path_to_samples[index]
                if path_to_samp.startswith('http'):
                    v = urlopen(path_to_samp).read()
                else:
                    with open(path_to_samp, 'rb') as f:
                        v = f.read()
                return len(v)
            except Exception as e:
                print(self._path_to_samples[index], e)
                return 0
        
        def __len__(self):
            return len(self._path_to_samples)
    
    dataset = VideoSet(list_file, base_dir)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=threads,
        pin_memory=False,
        drop_last=True,)
    return dataset, loader

    
def loader_all(list_file, base_dir, batch_size, threads):
    import loader
    return loader.train_loader(list_file, base_dir, batch_size, threads, hvd=hvd)


def main():
    
    threads = 128

    '''dataset, loader = loader_download(
        'train.txt',
        'http://filer.ai.yy.com:9889/dataset/heliangliang/imagenet/train/',
        #'val.txt',
        #'http://filer.ai.yy.com:9889/dataset/heliangliang/imagenet/val/',
        batch_size=100,
        threads=threads)'''

    dataset, loader = loader_all(
        'data/train.txt',
        #'http://filer.ai.yy.com:9889/dataset/heliangliang/imagenet/train/',
        #'http://10.29.1.2:9200/dataset/heliangliang/imagenet/train/',
        'http://10.29.4.248:9200/dataset/heliangliang/imagenet/train/',
        #'data/train2.txt',
        #'/data/local/imagenet/train2/',
        batch_size=100,
        threads=threads)
    print(hvd.rank(), len(dataset), len(loader))

    for k in range(10):
        ts = [0]
        t0 = time.time()
        i = -1
        for d in loader:
            i += 1
            #print('error', int(torch.sum(d==0)))
            print(d[0][1].shape)
            t1 = time.time()
            print('rank %s, k %s, i %s ---- t: %.4f (%.4f)' % (hvd.rank(), k, i, t1-t0, sum(ts)/len(ts)))
            ts.append(t1-t0)
            ts = ts[-threads:]
            t0 = t1


if __name__ == "__main__":
    main()


