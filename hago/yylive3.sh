

# 线上模型
# 场景，38类
'ckpt-yylive/scene_cls-5b-2/ckpt/checkpoint-iter-064000.pyth'
# 真人[0-3]、挂屏[4-5]、影视[6-10]、yy熊[11-12]、游戏[13-14]
'ckpt-yylive/multi_label-1c-fixres/ckpt/checkpoint-iter-008000.pyth'
# 游戏，8类
'ckpt-yylive/scene-game-2/ckpt/checkpoint-iter-102000.pyth'
# 人声，3类
'ckpt-yylive/uid_has_voice-6-fixres/ckpt/checkpoint-iter-010000.pyth'
# 过滤连麦模型(未上线)
'ckpt-yylive/proc_play-1c/ckpt/checkpoint-iter-008000.pyth'


# 新模型
# https://git.yy.com/aimodel/cv/yylive-content-understanding/sdk_deploy/-/blob/master/yylive-ai-scene_cls-7-fixres-checkpoint-iter-016000.pyth.bn


##################################################################################################################
# 多标签
#

# 20210901-0930
# 20211001-1031
# 20211101-1118
for ((dd=20211001; dd<=20211031; ++dd)); do
horovodrun -np 8 python hago/predict_hago.py \
--list-file "../yylive/search-same/img-list/all-$dd.txt.1" \
--root-dir "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/img-sdk/$dd/" \
--num-classes 15 \
--net "resnest50" \
--img-size 256 \
--batch-size 64 \
--ckpt "ckpt-yylive/multi_label-1c-fixres/ckpt/checkpoint-iter-008000.pyth" \
--out-per-n 1000000 \
--out "/data/local/pred/multi_label-1c-fixres-$dd"
done

dd=20211117
horovodrun -np 8 python hago/predict_hago.py \
--list-file "../yylive/search-same/img-list/all-$dd.txt.1" \
--root-dir "/data/local/search-same/$dd/" \
--num-classes 41 \
--net "resnest50" \
--img-size 256 \
--batch-size 64 \
--ckpt "ckpt-yylive/scene_cls-6b/ckpt/checkpoint-iter-064000.pyth" \
--out-per-n 1000000 \
--out "/data/local/pred/scene_cls-6b-$dd"

horovodrun -np 8 python hago/predict_hago.py \
--list-file "../yylive/search-same/img-list/all-$dd.txt.1" \
--root-dir "/data/local/search-same/$dd/" \
--num-classes 38 \
--net "resnest50" \
--img-size 256 \
--batch-size 64 \
--ckpt "ckpt-yylive/scene_cls-5b-2/ckpt/checkpoint-iter-064000.pyth" \
--out-per-n 1000000 \
--out "/data/local/pred/scene_cls-5b-2-$dd"

horovodrun -np 8 python hago/predict_hago.py \
--list-file "../yylive/search-same/img-list/all-$dd.txt.1" \
--root-dir "/data/local/search-same/$dd/" \
--num-classes 15 \
--net "resnest50" \
--img-size 256 \
--batch-size 64 \
--ckpt "ckpt-yylive/multi_label-1c-fixres/ckpt/checkpoint-iter-008000.pyth" \
--out-per-n 1000000 \
--out "/data/local/pred/multi_label-1c-fixres-$dd"

horovodrun -np 8 python hago/predict_hago.py \
--list-file "../yylive/search-same/img-list/all-$dd.txt.1" \
--root-dir "/data/local/search-same/$dd/" \
--num-classes 8 \
--net "resnest50" \
--img-size 256 \
--batch-size 64 \
--ckpt "ckpt-yylive/scene-game-2/ckpt/checkpoint-iter-102000.pyth" \
--out-per-n 1000000 \
--out "/data/local/pred/game-2-fixres-$dd"


--root-dir "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/img-sdk/$dd/" \
for ((dd=20211119; dd<=20211124; ++dd)); do
horovodrun -np 8 python hago/predict_hago.py \
--list-file "../yylive/search-same/img-list/all-$dd.txt.1" \
--root-dir "/data/local/search-same/$dd/" \
--num-classes 43 \
--net "resnest50" \
--img-size 256 \
--batch-size 64 \
--ckpt "ckpt-yylive/scene_cls-6b-2/ckpt/checkpoint-iter-064001.pyth" \
--out-per-n 1000000 \
--out "/data/local/pred/scene_cls-6b-2-$dd"
done



PREDICT() {
cd /data/remote/Moco/
horovodrun -np 8 python hago/predict_hago.py \
--list-file "$3" \
--root-dir "$4" \
--num-classes $2 \
--net "resnest50" \
--img-size 256 \
--batch-size 128 \
--ckpt "$1" \
--out-per-n 100000 \
--out "$5"
cd -
}
# ckpt-yylive/scene_cls-6b-2/ckpt/checkpoint-iter-064001.pyth  43
# ckpt-yylive/uid_has-3-pos-l/ckpt/checkpoint-iter-040000.pyth 3
# ckpt-yylive/multi_label-1c-fixres/ckpt/checkpoint-iter-008000.pyth 15
CKPT="ckpt-yylive/uid_has-3-pos-l/ckpt/checkpoint-iter-040000.pyth 3"
OUTN="uid_has-3-pos-l"
CKPT="ckpt-yylive/multi_label-1c-fixres/ckpt/checkpoint-iter-008000.pyth 15"
OUTN="multi_label-1c-fixres"
for ((dd=20210901; dd<=20210930; ++dd)); do
PREDICT $CKPT \
/data/remote/yylive/scene_cls/ds/$dd.txt /data/local/search-same/$dd/ \
/data/local/pred/$OUTN-$dd
done; \
for ((dd=20211001; dd<=20211031; ++dd)); do
PREDICT $CKPT \
/data/remote/yylive/scene_cls/ds/$dd.txt /data/local/search-same/$dd/ \
/data/local/pred/$OUTN-$dd
done; \
for ((dd=20211101; dd<=20211124; ++dd)); do
PREDICT $CKPT \
/data/remote/yylive/scene_cls/ds/$dd.txt /data/local/search-same/$dd/ \
/data/local/pred/$OUTN-$dd
done



horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/tra7-multi-soft.txt.1" \
--root-dir "/data/local/search-same/" \
--batch-size 64 \
--soft-label 1 \
--num-classes 43,3,2 \
--img-size 224 \
--base-lr 0.03 \
--lr-stages-step 96000,192000,256001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-7" \
--ckpt-save-interval 4000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/tra7-multi-soft.txt.1" \
--root-dir "/data/local/search-same/" \
--batch-size 64 \
--soft-label 1 \
--num-classes 43,3,2 \
--img-size 224 \
--base-lr 0.03 \
--lr-stages-step 96000,192000,256001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-5 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-7-2" \
--ckpt-save-interval 4000 \
--fp16 0


horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/tra7-multi-soft.txt.1" \
--root-dir "/data/local/search-same/" \
--batch-size 64 \
--soft-label 1 \
--num-classes 43,3,2 \
--img-size 256 \
--valloader 1 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/scene_cls-7/ckpt/checkpoint-iter-256000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-7-fine" \
--ckpt-save-interval 1000 \
--fp16 0


horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/tra7-multi-soft.txt.1" \
--root-dir "/data/local/search-same/" \
--batch-size 64 \
--soft-label 1 \
--num-classes 43,3,2 \
--img-size 256 \
--valloader 1 \
--base-lr 0.01 \
--lr-stages-step 2000,3000,4001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-5 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/scene_cls-7/ckpt/checkpoint-iter-256000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-7-fine-2" \
--ckpt-save-interval 500 \
--fp16 0


horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/tra7-multi-soft.txt.1" \
--root-dir "/data/local/search-same/" \
--batch-size 64 \
--soft-label 1 \
--num-classes 43,3,2 \
--img-size 256 \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/scene_cls-7/ckpt/checkpoint-iter-256000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-7-fixres" \
--ckpt-save-interval 1000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/tra7-multi-soft.txt.1" \
--root-dir "/data/local/search-same/" \
--batch-size 64 \
--soft-label 1 \
--num-classes 43,3,2 \
--img-size 256 \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-5 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/scene_cls-7-2/ckpt/checkpoint-iter-256000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-7-2-fixres" \
--ckpt-save-interval 1000 \
--fp16 0


horovodrun -np 8 python hago/train_hago_bn.py \
--list-file "../yylive/scene_cls/tra7-multi-soft.txt.1" \
--root-dir "/data/local/search-same/" \
--batch-size 64 \
--soft-label 1 \
--num-classes 43,3,2 \
--img-size 256 \
--valloader 1 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/scene_cls-7-fine-2/ckpt/checkpoint-iter-004000.pyth"


DIRNAME=ckpt-yylive/scene_cls-7
Nckpts=256000
Ninterval=4000
DIRNAME=ckpt-yylive/scene_cls-7-fine
Nckpts=16000
Ninterval=1000
DIRNAME=ckpt-yylive/scene_cls-7-fixres
Nckpts=16000
Ninterval=1000
for ((i=0; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    while [ ! -s $DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth ] ; do
        sleep 10s
    done
    horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../yylive/scene_cls/val6b1.txt.refine.lubo.1.local" \
    --root-dir "/data/local/scene_cls-val6b1/" \
    --num-classes 48 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth.bn" \
    --out-per-n 10000 \
    --out "$DIRNAME/val/val-scene_cls-bn-$part_n"
    horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../yylive/uid_has/val6.txt.pos.local" \
    --root-dir "/data/local/uid_has-val6/" \
    --num-classes 48 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth.bn" \
    --out-per-n 10000 \
    --out "$DIRNAME/val/val-uid_has-bn-$part_n"
done
for ((i=0; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    python hago/val_hago.py \
    "../yylive/scene_cls/val6b1.txt.refine.lubo.1" \
    "$DIRNAME/val/val-scene_cls-bn-$part_n" \
    "${DIRNAME}/log/val-scene_cls-bn/" \
    ${i} 43 0
    python hago/val_hago.py \
    "../yylive/uid_has/val6.txt.pos" \
    "$DIRNAME/val/val-uid_has-bn-$part_n" \
    "${DIRNAME}/log/val-uid_has-bn/" \
    ${i} 3 43
done


##################################################################################################################
# 人声模型
#
horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has_voice/tra.txt.1" \
--root-dir "/data/local/" \
--load-npy 1 \
--batch-size 64 \
--num-classes 3 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64001 \
--mixup 1.0 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_has_voice-uid_cls-4-fixres/ckpt/checkpoint-iter-064000.pyth" \
--ckpt-log-dir "ckpt-yylive/voice-1" \
--ckpt-save-interval 1000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has_voice/tra.txt.1" \
--root-dir "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/app/scene_cls2/" \
--load-npy 1 \
--batch-size 64 \
--num-classes 3 \
--img-size 256 \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 4000,6000,8001 \
--mixup 0 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/voice-1/ckpt/checkpoint-iter-064000.pyth" \
--ckpt-log-dir "ckpt-yylive/voice-1-fixres" \
--ckpt-save-interval 500 \
--fp16 0


DIRNAME=ckpt-yylive/voice-1
Nckpts=64000
Ninterval=1000
for ((i=64000; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    while [ ! -s $DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth ] ; do
        sleep 1m
    done
    sleep 2s
    horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../yylive/uid_has_voice/val.txt.split" \
    --root-dir "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/app/scene_cls2/" \
    --load-npy 1 \
    --num-classes 3 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    --out-per-n 10000 \
    --out "$DIRNAME/val/val-$part_n"
    python hago/val_hago.py \
    "../yylive/uid_has_voice/val.txt.split" \
    "$DIRNAME/val/val-$part_n" \
    "${DIRNAME}/log/val1/" \
    ${i} &
done



##################################################################################################################
# 真人模型
#
horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has/tra6.txt.1.pos" \
--root-dir "/data/local/mp4-m3u8/" \
--batch-size 64 \
--num-classes 3 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 24000,32000,40001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_has-3-pos-l" \
--ckpt-save-interval 1000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has/tra6.txt.1.pos" \
--root-dir "/data/local/mp4-m3u8/" \
--batch-size 64 \
--num-classes 3 \
--img-size 256 \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 2000,3000,4001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_has-3-pos/ckpt/checkpoint-iter-020000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_has-3-pos-fixres" \
--ckpt-save-interval 500 \
--fp16 0

DIRNAME=ckpt-yylive/uid_has-3-pos-l
Nckpts=40000
Ninterval=1000
for ((i=11000; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    while [ ! -s $DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth ] ; do
        sleep 10s
    done
    horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../yylive/uid_has/val6.txt.pos" \
    --root-dir "/data/local/mp4-m3u8/" \
    --num-classes 3 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    --out-per-n 10000 \
    --out "$DIRNAME/val/val-$part_n"
    python hago/val_hago.py \
    "../yylive/uid_has/val6.txt.pos" \
    "$DIRNAME/val/val-$part_n" \
    "${DIRNAME}/log/val/" \
    ${i} &
done


##################################################################################################################
# 场景模型
#

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/tra6.txt.1" \
--root-dir "/data/local/" \
--batch-size 64 \
--num-classes 42 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-6" \
--ckpt-save-interval 1000 \
--rand-corner 1 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/tra6b.txt.1" \
--root-dir "/data/local/,http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/" \
--batch-size 64 \
--num-classes 41 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-6b" \
--ckpt-save-interval 1000 \
--rand-corner 1 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/tra6b.txt.refine.lubo.1" \
--root-dir "/data/local/mp4-m3u8/,http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/" \
--batch-size 64 \
--num-classes 43 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-6b-2" \
--ckpt-save-interval 1000 \
--rand-corner 1 \
--fp16 0

DIRNAME=ckpt-yylive/scene_cls-6b-2
Nckpts=64000
Ninterval=4000
for ((i=0; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    while [ ! -s $DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth ] ; do
        sleep 10s
    done
    horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../yylive/scene_cls/val6b1.txt.refine.lubo.1.local" \
    --root-dir "/data/local/scene_cls-val6b1/" \
    --num-classes 43 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    --out-per-n 10000 \
    --out "$DIRNAME/val/val-$part_n"
    python hago/val_hago.py \
    "../yylive/scene_cls/val6b1.txt.refine.lubo" \
    "$DIRNAME/val/val-$part_n" \
    "${DIRNAME}/log/val/" \
    ${i} &
done

