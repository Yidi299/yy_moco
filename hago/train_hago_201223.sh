
# 清理
for dirname in pinlei-11* ; do
    echo $dirname
    cd $dirname/ckpt/
    for name in `ll | head -n -1 | awk '{if (NF>2) print $NF}'`; do
        rm -f $name
    done
    cd ../..
done


# noisy student数据计算
cat ../hago/daily-tag/data-train/pinlei-tra-1223.txt.2.local \
../hago/daily-tag/data-train/pinlei-tra-1110.txt.local | awk '{print $1}' | sort -u > \
../hago/daily-tag/data-train/pinlei-tra-1223.txt.2.local.u


for DIRNAME in \
pinlei-1223-moco1-init.2-sam \
pinlei-1223-moco1-init.2.1m-sam \
; do
RES=256
horovodrun -np 8 python hago/predict_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1223.txt.2.local.u" \
--root-dir  "/data/local/imgs-tra/" \
--num-classes 126 \
--net "resnest101" \
--img-size ${RES} \
--batch-size 128 \
--ckpt "ckpt-hago/${DIRNAME}/ckpt/checkpoint-iter-064000.pyth" \
--out-per-n 100 \
--out  "ckpt-hago/${DIRNAME}/340w/ret"
done
DIRNAME=pinlei-1223-moco1-init.2.1m-sam-200
RES=384
horovodrun -np 8 python hago/predict_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1223.txt.2.local.u" \
--root-dir  "/data/local/imgs-tra/" \
--num-classes 126 \
--net "resnest200" \
--img-size ${RES} \
--batch-size 128 \
--ckpt "ckpt-hago/${DIRNAME}/ckpt/checkpoint-iter-128000.pyth" \
--out-per-n 100 \
--out  "ckpt-hago/${DIRNAME}/340w/ret"


# 自动验证模型
# --list-file "../hago/daily-tag/data-val/pinlei-img-1224.txt"
# --list-file "../hago/daily-tag/data-val/pinlei-img-1224.txt.local"
# --root-dir  "http://filer.ai.yy.com:9899/dataset/"
# --root-dir  "http://10.28.32.58:9200/dataset/"
# --root-dir  "/data/local/imgs-val/"

DIRNAME=ckpt-hago/pinlei-1223-moco1-init.2-sam-soft-label-gt-200
RES=384
Nckpts=128
Ninterval=1
DIRNAME=ckpt-hago/pinlei-1223-moco1-init.2-sam-soft-label-gt-200-fixres-320-layer4
RES=448
Nckpts=8
Ninterval=1

for ((i=128; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%03d" ${i}`
    echo $i, 'wait'; while [ 1 ] ; do
        [[ -f "${DIRNAME}/ckpt/checkpoint-iter-${part_n}000.pyth" ]] && break
        sleep 1s
    done; sleep 1s; echo $i, 'begin'
    horovodrun -np 8 python hago/predict_hago.py \
        --list-file "../hago/daily-tag/data-val/pinlei-img-1224.txt" \
        --root-dir  "http://10.28.32.58:9200/dataset/" \
        --num-classes 126 \
        --net "resnest200" \
        --img-size ${RES} \
        --batch-size 32 \
        --ckpt "${DIRNAME}/ckpt/checkpoint-iter-${part_n}000.pyth" \
        --out  "${DIRNAME}/val/pinlei-img-1224.txt-${i}"
done
for ((i=128; i<=$Nckpts; i+=$Ninterval)); do
    echo $i, 'wait'; while [ 1 ] ; do
        [[ -f "${DIRNAME}/val/pinlei-img-1224.txt-${i}-pr.npy" ]] && break
        sleep 1s
    done; sleep 1s; echo $i, 'begin'
    python hago/val_hago.py \
        ../hago/daily-tag/data-val/pinlei-img-1224.txt \
        ${DIRNAME}/val/pinlei-img-1224.txt-${i} \
        ${DIRNAME}/log/val/ \
        ${i}000
done




# cat ../hago/daily-tag/data-train/pinlei-tra-1223.txt | awk -F"/" '{print $NF}' | awk '{print $1".jpg",$2}' > ../hago/daily-tag/data-train/pinlei-tra-1223.txt.local


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1223.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 126 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1223-moco1-init" \
--ckpt-save-interval 1000 \
--fp16 1

# 映射了搞笑类id
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1223.txt.2.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 126 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1223-moco1-init.2" \
--ckpt-save-interval 1000 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1223.txt.2.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 126 \
--sam 1 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1223-moco1-init.2-sam" \
--ckpt-save-interval 1000 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1223.txt.2.1m.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 126 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1223-moco1-init.2.1m" \
--ckpt-save-interval 1000 \
--fp16 1

### sam
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1223.txt.2.1m.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 126 \
--sam 1 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1223-moco1-init.2.1m-sam" \
--ckpt-save-interval 1000 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1223.txt.2.1m.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 126 \
--sam 1 \
--pretrained-ckpt "ckpt-hago/pinlei-1223-moco1-init.2.1m-sam/ckpt/checkpoint-iter-064000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16000 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1223-moco1-init.2.1m-sam-fixres-320-layer4" \
--ckpt-save-interval 1000 \
--fp16 1

# resnet-200
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1223.txt.2.1m.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 32 \
--num-classes 126 \
--net "resnest200" \
--sam 1 \
--pretrained-ckpt "ckpt-moco/resnest200-75117900.pth" \
--base-lr 0.01 \
--lr-stages-step 64000,96000,128000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1223-moco1-init.2.1m-sam-200" \
--ckpt-save-interval 2000 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1223.txt.2.1m.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 126 \
--net "resnest200" \
--sam 1 \
--pretrained-ckpt "ckpt-hago/pinlei-1223-moco1-init.2.1m-sam-200/ckpt/checkpoint-iter-128000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16000 \
--img-size 448 \
--ckpt-log-dir "ckpt-hago/pinlei-1223-moco1-init.2.1m-sam-200-fixres-320-layer4" \
--ckpt-save-interval 1000 \
--fp16 1







horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1223.txt.2.local.e" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 126 \
--soft-label-gt 1 \
--sam 1 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1223-moco1-init.2-sam-soft-label-gt" \
--ckpt-save-interval 1000 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1223.txt.2.local.e" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 32 \
--num-classes 126 \
--soft-label-gt 1 \
--net "resnest200" \
--sam 1 \
--pretrained-ckpt "ckpt-moco/resnest200-75117900.pth" \
--base-lr 0.01 \
--lr-stages-step 64000,96000,128000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1223-moco1-init.2-sam-soft-label-gt-200" \
--ckpt-save-interval 2000 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1223.txt.2.1m.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 126 \
--net "resnest200" \
--sam 1 \
--pretrained-ckpt "ckpt-hago/pinlei-1223-moco1-init.2-sam-soft-label-gt-200/ckpt/checkpoint-iter-128000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 4000,6000,8000 \
--img-size 448 \
--ckpt-log-dir "ckpt-hago/pinlei-1223-moco1-init.2-sam-soft-label-gt-200-fixres-320-layer4" \
--ckpt-save-interval 1000 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1223.txt.2.local.e" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 126 \
--soft-label 1 \
--sam 1 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1223-moco1-init.2-sam-soft-label" \
--ckpt-save-interval 1000 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1223.txt.2.1m.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 126 \
--net "resnest101" \
--sam 1 \
--pretrained-ckpt "ckpt-hago/pinlei-1223-moco1-init.2-sam-soft-label-gt/ckpt/checkpoint-iter-064000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16000 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1223-moco1-init.2-sam-soft-label-gt-fixres-320-layer4" \
--ckpt-save-interval 1000 \
--fp16 1




