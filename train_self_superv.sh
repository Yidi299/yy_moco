
# NaN
horovodrun -np 8 python train_self_superv.py \
--list-file "data/train.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/heliangliang/imagenet/train/" \
--net "resnet50" \
--alg "SimSiam" \
--batch-size 64 \
--base-lr 0.5 \
--lr-stages-step 100000,170000,230000,250010 \
--lr-warmup 0.1 \
--lr-warmup-step 5000 \
--weight-decay 1e-5 \
--img-size 224 \
--ckpt-log-dir "ckpt-SimSiam/imagenet-lr-0.5-epoch-100" \
--ckpt-save-interval 5000 \
--fp16 1


horovodrun -np 8 python train_self_superv.py \
--list-file "data/train.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/heliangliang/imagenet/train/" \
--net "resnet50" \
--alg "SimSiam" \
--batch-size 64 \
--base-lr 0.1 \
--lr-stages-step 100000,170000,230000,250010 \
--lr-warmup 0.1 \
--lr-warmup-step 5000 \
--weight-decay 1e-4 \
--no-bias-bn-wd 1 \
--img-size 224 \
--ckpt-log-dir "ckpt-SimSiam/imagenet-lr-0.1-epoch-100" \
--ckpt-save-interval 5000 \
--fp16 1



horovodrun -np 8 python train_self_superv.py \
--list-file "/data1/liuyidi/model/swin_V2/ssl/ssl_list.txt" \
--root-dir  "/data1/liuyidi/scene_cls/6b_2/" \
--net "swinv2_small_window8_256" \
--alg "SimSiam" \
--batch-size 16 \
--base-lr 0.1 \
--lr-stages-step 100000,170000,230000,250010 \
--lr-warmup 0.1 \
--lr-warmup-step 5000 \
--weight-decay 1e-4 \
--no-bias-bn-wd 1 \
--img-size 256 \
--ckpt-log-dir "/data1/liuyidi/model/swin_V2/ssl/cpkt" \
--ckpt-save-interval 5000 \
--fp16 1

