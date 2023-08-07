# 计算用于self-train的数据
# CKPTDIR="ckpt-hago/pinlei-1110-moco1-init-long-fixres-finetune-320-layer4"
# head -15000000 ../hago/daily-tag/img_list_all_20201122.txt > img_list_all_20201122.txt.1
# head -30000000 ../hago/daily-tag/img_list_all_20201122.txt|tail -15000000 > img_list_all_20201122.txt.2
# head -45000000 ../hago/daily-tag/img_list_all_20201122.txt|tail -15000000 > img_list_all_20201122.txt.3
# head -48000000 ../hago/daily-tag/img_list_all_20201122.txt|tail -3000000 > img_list_all_20201122.txt.4
head -500000 img_list_all_20201122.txt.4 | tail -500000 > img_list_all_20201122.txt.4.0
head -1000000 img_list_all_20201122.txt.4 | tail -500000 > img_list_all_20201122.txt.4.1
head -1500000 img_list_all_20201122.txt.4 | tail -500000 > img_list_all_20201122.txt.4.2
head -2000000 img_list_all_20201122.txt.4 | tail -500000 > img_list_all_20201122.txt.4.3
head -2500000 img_list_all_20201122.txt.4 | tail -500000 > img_list_all_20201122.txt.4.4
head -3000000 img_list_all_20201122.txt.4 | tail -500000 > img_list_all_20201122.txt.4.5


head -6000000  ../hago/daily-tag/img_list_all_20201122.txt|tail -6000000 > img_list_all_20201122.txt.1
head -12000000 ../hago/daily-tag/img_list_all_20201122.txt|tail -6000000 > img_list_all_20201122.txt.2
head -18000000 ../hago/daily-tag/img_list_all_20201122.txt|tail -6000000 > img_list_all_20201122.txt.3
head -24000000 ../hago/daily-tag/img_list_all_20201122.txt|tail -6000000 > img_list_all_20201122.txt.4
head -30000000 ../hago/daily-tag/img_list_all_20201122.txt|tail -6000000 > img_list_all_20201122.txt.5
head -36000000 ../hago/daily-tag/img_list_all_20201122.txt|tail -6000000 > img_list_all_20201122.txt.6
head -42000000 ../hago/daily-tag/img_list_all_20201122.txt|tail -6000000 > img_list_all_20201122.txt.7
head -48000000 ../hago/daily-tag/img_list_all_20201122.txt|tail -6000000 > img_list_all_20201122.txt.0

RES=448
CKPTDIR="ckpt-hago/pinlei-1110-imgnet-init-200-long-fixres-finetune-${RES}-layer4"
#CKPTDIR="ckpt-hago/pinlei-1110-moco1-init-long-320-fixres-finetune-${RES}-layer4"
for ((i=0; i<6; ++i)); do
CUDA_VISIBLE_DEVICES=$i python hago/predict_hago.py \
--list-file "img_list_all_20201122.txt.4.${i}" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/heliangliang/hago/" \
--num-classes 90 \
--net "resnest101" \
--img-size ${RES} \
--batch-size 32 \
--ckpt "${CKPTDIR}/ckpt/checkpoint-iter-016000.pyth" \
--out  "${CKPTDIR}/ret-4.${i}/img_list_all_20201122" \
--out-per-n 1000 &
done

for DIRNAME in \
pinlei-1110-moco1-init-long-fixres-finetune-320-layer4 \
pinlei-1110-self-init-340w-soft-fixres-320-layer4 \
pinlei-1110-self-add-tra-128k-soft-192k-fixres-320-layer4 \
pinlei-1110-self-init-340w-soft-gt-fixres-320-layer4 \
pinlei-1110-self-add-tra-128k-0.03-320-fixres-448-layer4 \
pinlei-1110-self-add-tra-128k-0.03-224k-fixres-320-layer4-2 \
pinlei-1110-self-init-128k-0.03-ft-gt-fixres-320-layer4 \
; do
RES=`echo $DIRNAME|awk -F"-layer4" '{print $1}'|awk -F"-" '{print $NF}'`
horovodrun -np 8 python hago/predict_hago.py \
--list-file "pinlei-tra-1110.txt.local.u" \
--root-dir  "/data/local/imgs-tra/" \
--num-classes 90 \
--net "resnest101" \
--img-size ${RES} \
--batch-size 128 \
--ckpt "ckpt-hago/${DIRNAME}/ckpt/checkpoint-iter-016000.pyth" \
--out-per-n 100 \
--out  "ckpt-hago/${DIRNAME}/340w/ret"
done

pinlei-1110-self-init-340w-soft-gt-200-fixres-448-layer4
resnest200


for M in m1 m2 m3 e1 e2; do
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.self.local.$M" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--soft-label-gt 1 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-340w-teacher-$M" \
--ckpt-save-interval 2000 \
--fp16 1
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-init-340w-teacher-$M/ckpt/checkpoint-iter-032000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-340w-teacher-$M-fixres-320-layer4" \
--ckpt-save-interval 1000 \
--fp16 1
done

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.self.local.e2" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--soft-label-gt 1 \
--sam 1 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 64000,96000,128001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-340w-teacher-e2-sam" \
--ckpt-save-interval 1000 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--batch-size 64 \
--num-classes 90 \
--sam 1 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.03 \
--lr-stages-step 64000,96000,128001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-long-sam" \
--ckpt-save-interval 1000 \
--fp16 1

DIR=ckpt-hago/pinlei-1110-self-init-340w-teacher-e2-sam
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--batch-size 64 \
--num-classes 90 \
--sam 1 \
--pretrained-ckpt "${DIR}/ckpt/checkpoint-iter-128000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 16000,24000,32001 \
--img-size 320 \
--ckpt-log-dir "${DIR}-fixres-320-layer4-sam-32k" \
--ckpt-save-interval 1000 \
--fp16 1



# 自动验证模型
# --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt"
# --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt.local"
# --root-dir  "http://filer.ai.yy.com:9899/dataset/"
# --root-dir  "http://10.28.32.58:9200/dataset/"
# --root-dir  "/data/local/imgs-val/"

DIRNAME=ckpt-hago/pinlei-1110-self-init-340w-teacher-e2-sam-222
RES=256
Nckpts=128
Ninterval=1
DIRNAME=ckpt-hago/pinlei-1110-self-init-340w-teacher-e2-sam-fixres-320-layer4-sam-222
RES=320
Nckpts=16
Ninterval=1

for ((i=16; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%03d" ${i}`
    echo $i, 'wait'; while [ 1 ] ; do
        [[ -f "${DIRNAME}/ckpt/checkpoint-iter-${part_n}000.pyth" ]] && break
        sleep 1s
    done; sleep 1s; echo $i, 'begin'
    horovodrun -np 8 python hago/predict_hago.py \
        --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt" \
        --root-dir  "http://10.28.32.58:9200/dataset/" \
        --num-classes 90 \
        --net "resnest101" \
        --img-size ${RES} \
        --batch-size 32 \
        --ckpt "${DIRNAME}/ckpt/checkpoint-iter-${part_n}000.pyth" \
        --out  "${DIRNAME}/val/pinlei-img-1112.txt-${i}"
done
for ((i=16; i<=$Nckpts; i+=$Ninterval)); do
    echo $i, 'wait'; while [ 1 ] ; do
        [[ -f "${DIRNAME}/val/pinlei-img-1112.txt-${i}-pr.npy" ]] && break
        sleep 1s
    done; sleep 1s; echo $i, 'begin'
    python hago/val_hago.py \
        ../hago/daily-tag/data-val/pinlei-img-1112.txt \
        ${DIRNAME}/val/pinlei-img-1112.txt-${i} \
        ${DIRNAME}/log/val/ \
        ${i}000
done


# 在340w上使用self标签实验
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.self.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-340w-hard" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-init-340w-hard/ckpt/checkpoint-iter-032000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-340w-hard-fixres-320-layer4" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-add-tra-128k-soft-192k/ckpt/checkpoint-iter-192000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-add-tra-128k-soft-192k-fixres-320-layer4" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.self-1.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-1-init-340w-hard" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-1-init-340w-hard/ckpt/checkpoint-iter-032000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-1-init-340w-hard-fixres-320-layer4" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.self.soft.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--soft-label 1 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-340w-soft-long" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-init-340w-soft-long/ckpt/checkpoint-iter-064000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-340w-soft-long-fixres-320-layer4" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.self.soft-gt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--soft-label-gt 1 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-340w-soft-gt" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.self.soft-gt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 32 \
--num-classes 90 \
--net "resnest200" \
--soft-label-gt 1 \
--pretrained-ckpt "ckpt-moco/resnest200-75117900.pth" \
--base-lr 0.01 \
--lr-stages-step 64000,96000,128001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-340w-soft-gt-200" \
--ckpt-save-interval 400 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--net "resnest200" \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-init-340w-soft-gt-200/ckpt/checkpoint-iter-128000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 448 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-340w-soft-gt-200-fixres-448-layer4" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.self.soft.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--soft-label 1 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-340w-soft+gt" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.self.soft.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--soft-label 1 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-init-340w-soft/ckpt/checkpoint-iter-032000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-340w-soft-fixres-320-layer4-soft" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.self.soft" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 90 \
--soft-label 1 \
--soft-label-t 0.5 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-340w-soft-t-0.5" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.self.soft.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--soft-label 1 \
--soft-label-t 2 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-340w-soft-t-2" \
--ckpt-save-interval 100 \
--fp16 1

# noisy student, 只用模型predict结果+监督数据部分，比例为约4:1
horovodrun -np 8 python hago/train_hago.py \
--list-file "pinlei-self_4800w-tra_340w-1110.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/heliangliang/hago/" \
--donot-shuffle 1 \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.03 \
--lr-stages-step 64000,96000,128001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03" \
--ckpt-save-interval 400 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "pinlei-self_4800w-tra_340w-1110.txt.soft" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/heliangliang/hago/" \
--donot-shuffle 1 \
--batch-size 64 \
--num-classes 90 \
--soft-label 1 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.03 \
--lr-stages-step 64000,96000,128001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-add-tra-128k-soft" \
--ckpt-save-interval 400 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "pinlei-self_4800w-tra_340w-1110.txt.soft" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/heliangliang/hago/" \
--donot-shuffle 1 \
--batch-size 64 \
--num-classes 90 \
--soft-label 1 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.03 \
--lr-stages-step 64000,128000,160000,192001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-add-tra-128k-soft-192k" \
--ckpt-save-interval 400 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "pinlei-self_4800w-tra_340w-1110.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/heliangliang/hago/" \
--donot-shuffle 1 \
--batch-size 32 \
--num-classes 90 \
--net "resnest101" \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.03 \
--lr-stages-step 96000,192000,256001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03-320" \
--ckpt-save-interval 800 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03/ckpt/checkpoint-iter-128000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03-fixres-320-layer4" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-init-340w-soft-gt/ckpt/checkpoint-iter-064000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-340w-soft-gt-fixres-320-layer4" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-add-tra-128k-soft/ckpt/checkpoint-iter-128000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-add-tra-128k-soft-fixres-320-layer4" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03-320/ckpt/checkpoint-iter-256000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 448 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03-320-fixres-448-layer4" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03/ckpt/checkpoint-iter-128000.pyth" \
--fixres 1 \
--base-lr 0.003 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03-fixres-320-layer4-0.003" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03/ckpt/checkpoint-iter-128000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 16000,32000,40000,48001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03-fixres-320-layer4-48k" \
--ckpt-save-interval 200 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03/ckpt/checkpoint-iter-128000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,16000,20000,24001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03-fixres-320-layer4-24k" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "pinlei-self_4800w-tra_340w-1110.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/heliangliang/hago/" \
--donot-shuffle 1 \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.03 \
--lr-stages-step 64000,128000,192000,224001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03-224k" \
--ckpt-save-interval 400 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03-224k/ckpt/checkpoint-iter-220000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 6000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03-224k-fixres-320-layer4" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03-224k/ckpt/checkpoint-iter-220000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03-224k-fixres-320-layer4-2" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-moco1-init-long-2/ckpt/checkpoint-iter-160000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-long-2-fixres-320-layer4" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-moco1-init-long-3/ckpt/checkpoint-iter-144000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-long-3-fixres-320-layer4" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03-224k/ckpt/checkpoint-iter-220000.pyth" \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03-224k-ft-gt" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03-224k-ft-gt/ckpt/checkpoint-iter-016000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-add-tra-128k-0.03-224k-ft-gt-fixres-320-layer4" \
--ckpt-save-interval 100 \
--fp16 1

# noisy student, 只用模型predict结果，统计显示，340有监督数据的predict结果的acc是81%，和训练曲线一致
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-self-learn-1110.txt.65536000" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/heliangliang/hago/" \
--donot-shuffle 1 \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.03 \
--lr-stages-step 64000,96000,128001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-128k-0.03" \
--ckpt-save-interval 400 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-init-128k-0.03/ckpt/checkpoint-iter-124000.pyth" \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-128k-0.03-ft-gt" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-init-128k-0.03-ft-gt/ckpt/checkpoint-iter-032000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-128k-0.03-ft-gt-fixres-320-layer4" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-init-128k-0.03/ckpt/checkpoint-iter-128000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-self-init-128k-0.03-fixres-320-layer4" \
--ckpt-save-interval 100 \
--fp16 1



# 实验结果表明2最优，但差别很小
# acc 0.7852837164443015 pr_xeqy_w_avg 0.7995056355891053 pr_xeqy_avg 0.49660893550643037
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-moco1-init-long/ckpt/checkpoint-iter-032000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-long-fixres-finetune-320-layer4-para-1" \
--ckpt-save-interval 100 \
--fp16 1

# BEST
# acc 0.7880364893104003 pr_xeqy_w_avg 0.8009132207337011 pr_xeqy_avg 0.495876730825032
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-moco1-init-long/ckpt/checkpoint-iter-032000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-long-fixres-finetune-320-layer4-para-2" \
--ckpt-save-interval 100 \
--fp16 1

# acc 0.786629963028452 pr_xeqy_w_avg 0.7999580649612013 pr_xeqy_avg 0.4974168806037163
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-moco1-init-long/ckpt/checkpoint-iter-032000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 5e-4 \
--lr-stages-step 8000,12000,16001 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-long-fixres-finetune-320-layer4-para-3" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/" \
    --num-classes 90 \
    --net "resnest101" \
    --img-size 320 \
    --batch-size 32 \
    --ckpt "ckpt-hago/pinlei-1110-moco1-init-long-fixres-finetune-320-layer4-para-3/ckpt/checkpoint-iter-016000.pyth" \
    --out  "ckpt-hago/pinlei-1110-moco1-init-long-fixres-finetune-320-layer4-para-3/val/pinlei-img-1112.txt-16"

python hago/val_hago.py \
    ../hago/daily-tag/data-val/pinlei-img-1112.txt \
    ckpt-hago/pinlei-1110-moco1-init-long-fixres-finetune-320-layer4-para-1/val/pinlei-img-1112.txt-16 \
    null 0

# dropout，wd，no-bias-bn-wd等参数不一致啊！！！
CKPTDIR=ckpt-hago/pinlei-1110-moco1-init-before-0
RES=320
CKPTDIR=ckpt-hago/pinlei-1110-imgnet-init-200-long
RES=448
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--batch-size 64 \
--num-classes 90 \
--net "resnest200" \
--pretrained-ckpt "${CKPTDIR}/ckpt/checkpoint-iter-064000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16100 \
--img-size $RES \
--ckpt-log-dir "${CKPTDIR}-fixres-finetune-$RES-layer4" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/" \
    --num-classes 90 \
    --net "resnest200" \
    --img-size $RES \
    --batch-size 32 \
    --ckpt "${CKPTDIR}-fixres-finetune-$RES-layer4/ckpt/checkpoint-iter-016000.pyth" \
    --out  "${CKPTDIR}-fixres-finetune-$RES-layer4/val/pinlei-img-1112.txt-16"

python hago/val_hago.py \
    ../hago/daily-tag/data-val/pinlei-img-1112.txt \
    ${CKPTDIR}-fixres-finetune-$RES-layer4/val/pinlei-img-1112.txt-16 \
    null 0


#=====================================================================================================
# 同样配置训练5个模型，看下验证集合波动情况
for iii in 1 2 3 4 5; do
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-$iii" \
--ckpt-save-interval 100 \
--fp16 1
done

# ${DIRNAME}/log/val直接用logits计算了pr，val-2修复了这个bug，修复后的曲线和原来趋势一致但高了4个多点
for nnn in 1 2 3 4 5; do
DIRNAME=ckpt-hago/pinlei-1110-moco1-init-$nnn
for ((iii=0; iii<=16; ++iii)); do
part_n=`printf "%03d" ${iii}` 
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/" \
    --num-classes 90 \
    --ckpt "${DIRNAME}/ckpt/checkpoint-iter-${part_n}000.pyth" \
    --out  "${DIRNAME}/val/pinlei-img-1112.txt-${iii}"
python hago/val_hago.py \
    ../hago/daily-tag/data-val/pinlei-img-1112.txt \
    ${DIRNAME}/val/pinlei-img-1112.txt-${iii} \
    ${DIRNAME}/log/val-2/ \
    ${iii}000
done
done

# 更长的训练时间
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-long" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.03 \
--lr-stages-step 48000,96000,128000,160001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-long-2" \
--ckpt-save-interval 400 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.local" \
--root-dir  "/data/local/imgs-tra/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.03 \
--lr-stages-step 96000,112000,128000,144001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-long-3" \
--ckpt-save-interval 400 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-long" \
--ckpt-save-interval 100 \
--fp16 1

# 320分辨率
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-long-320" \
--ckpt-save-interval 100 \
--fp16 1


# fixres
for RES in 224 256 288 320 352 384 416; do
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-moco1-init-long/ckpt/checkpoint-iter-032000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16100 \
--img-size $RES \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-long-fixres-finetune-$RES-layer4" \
--ckpt-save-interval 100 \
--fp16 1
done

for RES in 320 352 384 416 448 480 512; do
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-moco1-init-long-320/ckpt/checkpoint-iter-032000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size $RES \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-long-320-fixres-finetune-$RES-layer4" \
--ckpt-save-interval 100 \
--fp16 1
done

RES=416 #################################################################################
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-moco1-init-long-320/ckpt/checkpoint-iter-032000.pyth" \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size $RES \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-long-320-fixres-finetune-$RES-layer4-0.01-16k" \
--ckpt-save-interval 100 \
--fp16 1 #################################################################################

RES=416
DIRNAME=ckpt-hago/pinlei-1110-moco1-init-long-320-fixres-finetune-$RES-layer4-0.003-13k
mkdir ${DIRNAME}/val/
for ((i=0; i<=9; ++i)); do
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/" \
    --num-classes 90 \
    --img-size $RES \
    --batch-size 8 \
    --ckpt "${DIRNAME}/ckpt/checkpoint-iter-00${i}000.pyth" \
    --out  "${DIRNAME}/val/pinlei-img-1112.txt-${i}"
done
for ((i=0; i<=9; ++i)); do
python hago/val_hago.py \
    ../hago/daily-tag/data-val/pinlei-img-1112.txt \
    ${DIRNAME}/val/pinlei-img-1112.txt-${i} \
    ${DIRNAME}/log/val ${i}000
done

DIRNAME=ckpt-hago/pinlei-1110-moco1-init-long-320
mkdir ${DIRNAME}/val/
for RES in 224 256 288 320 352 384 416 448 480 512; do
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/" \
    --num-classes 90 \
    --img-size $RES \
    --batch-size 64 \
    --ckpt "${DIRNAME}/ckpt/checkpoint-iter-032000.pyth" \
    --out  "${DIRNAME}/val/pinlei-img-1112.txt-fixres-$RES"
done
for RES in 224 256 288 320 352 384 416 448 480 512; do
echo "RES:$RES"
python hago/val_hago.py \
    ../hago/daily-tag/data-val/pinlei-img-1112.txt \
    ${DIRNAME}/val/pinlei-img-1112.txt-fixres-$RES \
    null 0
done

DIRNAME=ckpt-hago/pinlei-1110-moco1-init-long-320
for RES in 224 256 288 320 352 384 416 448 480 512; do
mkdir ${DIRNAME}-fixres-finetune-$RES/val/
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/" \
    --num-classes 90 \
    --img-size $RES \
    --batch-size 64 \
    --ckpt "${DIRNAME}-fixres-finetune-$RES/ckpt/checkpoint-iter-005000.pyth" \
    --out  "${DIRNAME}-fixres-finetune-$RES/val/pinlei-img-1112.txt-5"
done
for RES in 224 256 288 320 352 384 416 448 480 512; do
echo "RES:$RES"
python hago/val_hago.py \
    ../hago/daily-tag/data-val/pinlei-img-1112.txt \
    ${DIRNAME}-fixres-finetune-$RES/val/pinlei-img-1112.txt-5 \
    null 0
done

# 对比imgnet初始化
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/resnest101-22405ba7.pth" \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-imgnet-init-1" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--num-classes 90 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-moco/resnest50-528c19ca.pth" \
--batch-size 64 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-imgnet-init-50" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--num-classes 90 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest101" \
--pretrained-ckpt "ckpt-moco/resnest101-22405ba7.pth" \
--batch-size 64 \
--img-size 256 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-imgnet-init-101" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--num-classes 90 \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest200" \
--pretrained-ckpt "ckpt-moco/resnest200-75117900.pth" \
--batch-size 32 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-imgnet-init-200" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--num-classes 90 \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest200" \
--pretrained-ckpt "ckpt-moco/resnest200-75117900.pth" \
--batch-size 32 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-imgnet-init-200-long" \
--ckpt-save-interval 200 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--num-classes 90 \
--base-lr 0.01 \
--lr-stages-step 22000,32000,43100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest269" \
--pretrained-ckpt "ckpt-moco/resnest269-0cc87c48.pth" \
--batch-size 24 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-imgnet-init-269" \
--ckpt-save-interval 100 \
--fp16 1


# 数据量采样为1/2^n，看效果变化
cd ../hago/daily-tag/; cat data-train/pinlei-tra-1110.txt| sort -u > pinlei-tra-1110.txt.u
python
>>> lines = open('pinlei-tra-1110.txt.u').readlines()
>>> lines = [i.strip() for i in lines]
>>> import random
>>> random.shuffle(lines)
>>> f = open('pinlei-tra-1110.txt.u.shuffle', 'w')
>>> for i in lines: print(i, file=f) 
... 
>>> exit()
head -1705472 pinlei-tra-1110.txt.u.shuffle > pinlei-tra-1110.txt.u.shuffle.1
head -852736 pinlei-tra-1110.txt.u.shuffle > pinlei-tra-1110.txt.u.shuffle.2
head -426368 pinlei-tra-1110.txt.u.shuffle > pinlei-tra-1110.txt.u.shuffle.3
head -213184 pinlei-tra-1110.txt.u.shuffle > pinlei-tra-1110.txt.u.shuffle.4
head -106592 pinlei-tra-1110.txt.u.shuffle > pinlei-tra-1110.txt.u.shuffle.5

for iii in 1 2 3 4 5; do
python mk_data_balance.py pinlei-tra-1110.txt.u.shuffle.$iii data-train/pinlei-tra-1110.txt.half.$iii
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.half.$iii" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-half-$iii" \
--ckpt-save-interval 100 \
--fp16 1
done

# 取一个月样本，时间差1、2、4、8周，看效果差异
0 1110 1011 770305
1 1103 1004 777194
2 1027 0927 822811
3 1013 0913 1052810
4 0915 0816 1224753

for iii in 0 1 2 3 4; do
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-1110.txt.before.$iii" \
--root-dir  "http://10.28.32.58:9200/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1110-moco1-init-before-$iii" \
--ckpt-save-interval 100 \
--fp16 1
done





# 计算验证结果
# imagenet初始化
DIRNAME=ckpt-hago/pinlei-1110-imgnet-init-1
mkdir ${DIRNAME}/val/
for ((iii=0; iii<=16; ++iii)); do
part_n=`printf "%03d" ${iii}` 
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/" \
    --num-classes 90 \
    --ckpt "${DIRNAME}/ckpt/checkpoint-iter-${part_n}000.pyth" \
    --out  "${DIRNAME}/val/pinlei-img-1112.txt-${iii}"
python hago/val_hago.py \
    ../hago/daily-tag/data-val/pinlei-img-1112.txt \
    ${DIRNAME}/val/pinlei-img-1112.txt-${iii} \
    ${DIRNAME}/log/val/ \
    ${iii}000
done

DIRNAME=ckpt-hago/pinlei-1110-imgnet-init-50
mkdir ${DIRNAME}/val/
for ((iii=0; iii<=16; ++iii)); do
part_n=`printf "%03d" ${iii}` 
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/" \
    --num-classes 90 \
    --net "resnest50" \
    --img-size 256 \
    --ckpt "${DIRNAME}/ckpt/checkpoint-iter-${part_n}000.pyth" \
    --out  "${DIRNAME}/val/pinlei-img-1112.txt-${iii}"
python hago/val_hago.py \
    ../hago/daily-tag/data-val/pinlei-img-1112.txt \
    ${DIRNAME}/val/pinlei-img-1112.txt-${iii} \
    ${DIRNAME}/log/val/ \
    ${iii}000
done

DIRNAME=ckpt-hago/pinlei-1110-imgnet-init-101
mkdir ${DIRNAME}/val/
for ((iii=0; iii<=16; ++iii)); do
part_n=`printf "%03d" ${iii}` 
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/" \
    --num-classes 90 \
    --net "resnest101" \
    --img-size 304 \
    --ckpt "${DIRNAME}/ckpt/checkpoint-iter-${part_n}000.pyth" \
    --out  "${DIRNAME}/val/pinlei-img-1112.txt-${iii}"
python hago/val_hago.py \
    ../hago/daily-tag/data-val/pinlei-img-1112.txt \
    ${DIRNAME}/val/pinlei-img-1112.txt-${iii} \
    ${DIRNAME}/log/val/ \
    ${iii}000
done

DIRNAME=ckpt-hago/pinlei-1110-imgnet-init-200
mkdir ${DIRNAME}/val/
for ((iii=0; iii<=32; ++iii)); do
part_n=`printf "%03d" ${iii}` 
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/" \
    --num-classes 90 \
    --net "resnest200" \
    --img-size 384 \
    --batch-size 64 \
    --ckpt "${DIRNAME}/ckpt/checkpoint-iter-${part_n}000.pyth" \
    --out  "${DIRNAME}/val/pinlei-img-1112.txt-${iii}"
python hago/val_hago.py \
    ../hago/daily-tag/data-val/pinlei-img-1112.txt \
    ${DIRNAME}/val/pinlei-img-1112.txt-${iii} \
    ${DIRNAME}/log/val/ \
    ${iii}000
done

DIRNAME=ckpt-hago/pinlei-1110-imgnet-init-269
mkdir ${DIRNAME}/val/
for ((iii=0; iii<=43; ++iii)); do
part_n=`printf "%03d" ${iii}` 
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/" \
    --num-classes 90 \
    --net "resnest269" \
    --img-size 384 \
    --batch-size 32 \
    --ckpt "${DIRNAME}/ckpt/checkpoint-iter-${part_n}000.pyth" \
    --out  "${DIRNAME}/val/pinlei-img-1112.txt-${iii}"
python hago/val_hago.py \
    ../hago/daily-tag/data-val/pinlei-img-1112.txt \
    ${DIRNAME}/val/pinlei-img-1112.txt-${iii} \
    ${DIRNAME}/log/val/ \
    ${iii}000
done


# 数据量采样
for nnn in 1 2 3 4 5; do
DIRNAME=ckpt-hago/pinlei-1110-moco1-init-half-$nnn
mkdir ${DIRNAME}/val/
for ((iii=0; iii<=16; ++iii)); do
part_n=`printf "%03d" ${iii}` 
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/" \
    --num-classes 90 \
    --ckpt "${DIRNAME}/ckpt/checkpoint-iter-${part_n}000.pyth" \
    --out  "${DIRNAME}/val/pinlei-img-1112.txt-${iii}"
python hago/val_hago.py \
    ../hago/daily-tag/data-val/pinlei-img-1112.txt \
    ${DIRNAME}/val/pinlei-img-1112.txt-${iii} \
    ${DIRNAME}/log/val/ \
    ${iii}000
done
done

#时间窗口采样
for nnn in 0 1 2 3 4; do
DIRNAME=ckpt-hago/pinlei-1110-moco1-init-before-$nnn
mkdir ${DIRNAME}/val/
for ((iii=0; iii<=16; ++iii)); do
part_n=`printf "%03d" ${iii}` 
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/" \
    --num-classes 90 \
    --ckpt "${DIRNAME}/ckpt/checkpoint-iter-${part_n}000.pyth" \
    --out  "${DIRNAME}/val/pinlei-img-1112.txt-${iii}"
python hago/val_hago.py \
    ../hago/daily-tag/data-val/pinlei-img-1112.txt \
    ${DIRNAME}/val/pinlei-img-1112.txt-${iii} \
    ${DIRNAME}/log/val/ \
    ${iii}000
done
done

DIRNAME=ckpt-hago/pinlei-1110-moco1-init-long
mkdir ${DIRNAME}/val/
for ((iii=0; iii<=32; ++iii)); do
part_n=`printf "%03d" ${iii}` 
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/" \
    --num-classes 90 \
    --ckpt "${DIRNAME}/ckpt/checkpoint-iter-${part_n}000.pyth" \
    --out  "${DIRNAME}/val/pinlei-img-1112.txt-${iii}"
python hago/val_hago.py \
    ../hago/daily-tag/data-val/pinlei-img-1112.txt \
    ${DIRNAME}/val/pinlei-img-1112.txt-${iii} \
    ${DIRNAME}/log/val/ \
    ${iii}000
done

DIRNAME=ckpt-hago/pinlei-1110-moco1-init-long-320
mkdir ${DIRNAME}/val/
for ((iii=0; iii<=32; ++iii)); do
part_n=`printf "%03d" ${iii}` 
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/pinlei-img-1112.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/" \
    --num-classes 90 \
    --img-size 384 \
    --ckpt "${DIRNAME}/ckpt/checkpoint-iter-${part_n}000.pyth" \
    --out  "${DIRNAME}/val/pinlei-img-1112.txt-${iii}"
python hago/val_hago.py \
    ../hago/daily-tag/data-val/pinlei-img-1112.txt \
    ${DIRNAME}/val/pinlei-img-1112.txt-${iii} \
    ${DIRNAME}/log/val/ \
    ${iii}000
done


