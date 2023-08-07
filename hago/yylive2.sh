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


##################################################################################################################
# 真人模型
#
horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has/tra6.txt.1" \
--root-dir "/data/local/" \
--batch-size 64 \
--num-classes 3 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_has-3" \
--ckpt-save-interval 1000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has/tra6.txt.1" \
--root-dir "/data/local/" \
--batch-size 64 \
--num-classes 3 \
--img-size 256 \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 1000,1500,2001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_has-3/ckpt/checkpoint-iter-016000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_has-3-fixres" \
--ckpt-save-interval 500 \
--fp16 0

DIRNAME=ckpt-yylive/uid_has-3
Nckpts=16000
Ninterval=1000
for ((i=0; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    while [ ! -s $DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth ] ; do
        sleep 1m
    done
    sleep 10s
    horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../yylive/uid_has/val6.txt" \
    --root-dir "/data/local/" \
    --num-classes 3 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    --out-per-n 10000 \
    --out "$DIRNAME/val/val-$part_n"
    python hago/val_hago.py \
    "../yylive/uid_has/val6.txt" \
    "$DIRNAME/val/val-$part_n" \
    "${DIRNAME}/log/val1/" \
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
--list-file "../yylive/scene_cls/tra4.txt.1" \
--root-dir "/data/local/" \
--batch-size 64 \
--num-classes 55 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-4" \
--ckpt-save-interval 1000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/tra5.txt.1" \
--root-dir "/data/local/,http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/" \
--batch-size 64 \
--num-classes 38 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-5" \
--ckpt-save-interval 1000 \
--rand-corner 1 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/tra5.txt.1" \
--root-dir "/data/local/,http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/" \
--batch-size 64 \
--num-classes 38 \
--img-size 256 \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 4000,6000,8001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/scene_cls-5/ckpt/checkpoint-iter-064000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-5-fixres" \
--ckpt-save-interval 1000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/data-wuxiao-jpg-lubo.txt" \
--root-dir "/data/local/" \
--batch-size 64 \
--num-classes 2 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 500,800,1001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/scene_cls-5/ckpt/checkpoint-iter-064000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-5-lubo" \
--ckpt-save-interval 100 \
--rand-corner 1 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/tra5b.txt.1" \
--root-dir "/data/local/,http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/" \
--batch-size 64 \
--num-classes 38 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-5b-2" \
--ckpt-save-interval 1000 \
--rand-corner 1 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/tra4.txt.1" \
--root-dir "/data/local/" \
--batch-size 64 \
--num-classes 55 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-4rc-2" \
--ckpt-save-interval 1000 \
--rand-corner 1 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/tra4.txt.1" \
--root-dir "/data/local/" \
--batch-size 128 \
--num-classes 55 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 64000,96000,128001 \
--sam 1 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnet18" \
--ckpt-log-dir "ckpt-yylive/scene_cls-4rc-resnet18" \
--ckpt-save-interval 1000 \
--rand-corner 1 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/tra4.txt.1" \
--root-dir "/data/local/" \
--batch-size 64 \
--num-classes 55 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-4rc-wh-ratio" \
--ckpt-save-interval 1000 \
--rand-corner 1 \
--loader-keep-wh-ratio 1 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/scene_cls/tra4.txt.1" \
--root-dir "/data/local/" \
--batch-size 64 \
--num-classes 55 \
--img-size 256 \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 2000,4000,6001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/scene_cls-4rc/ckpt/checkpoint-iter-032000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene_cls-4rc-fixres" \
--ckpt-save-interval 1000 \
--fp16 0

DIRNAME=ckpt-yylive/scene_cls-6b
Nckpts=64000
Ninterval=1000
for ((i=0; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../yylive/scene_cls/val6b1.txt" \
    --root-dir "/data/local/" \
    --num-classes 41 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    --out-per-n 10000 \
    --out "$DIRNAME/val/val-$part_n"
    python hago/val_hago.py \
    "../yylive/scene_cls/val6b1.txt" \
    "$DIRNAME/val/val-$part_n" \
    "${DIRNAME}/log/val1/" \
    ${i} &
done


# -------------------- test ----------------------------
python hago/val_hago-scene_cls.py ../yylive/scene_cls/val4.txt ckpt-yylive/scene_cls-4rc/val/val4-032000
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "/data/local/sdk-img/20210812-2.txt" \
    --root-dir "/data/local/sdk-img/20210812-2/" \
    --num-classes 55 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "ckpt-yylive/scene_cls-4rc/ckpt/checkpoint-iter-032000.pyth" \
    --out-per-n 10000 \
    --out "sdk-img-20210812-2-scene_cls-4rc-32k"
python hago/val_hago-scene_cls.py ../yylive/scene_cls/val5.txt ckpt-yylive/scene_cls-5/val/val-064000
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "/data/local/sdk-img/20210812-2.txt" \
    --root-dir "/data/local/sdk-img/20210812-2/" \
    --num-classes 38 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "ckpt-yylive/scene_cls-5/ckpt/checkpoint-iter-064000.pyth" \
    --out-per-n 10000 \
    --out "sdk-img-20210812-2-scene_cls-5-64k"
python hago/val_hago-scene_cls.py ../yylive/scene_cls/val5.txt ckpt-yylive/scene_cls-5-fixres/val/val-008000
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "/data/local/sdk-img/20210812-2.txt" \
    --root-dir "/data/local/sdk-img/20210812-2/" \
    --num-classes 38 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "ckpt-yylive/scene_cls-5-fixres/ckpt/checkpoint-iter-008000.pyth" \
    --out-per-n 10000 \
    --out "sdk-img-20210812-2-scene_cls-5-fixres-8k"
python hago/val_hago-scene_cls.py ../yylive/scene_cls/val5.txt ckpt-yylive/scene_cls-5b-2/val/val-064000
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "/data/local/sdk-img/20210812-2.txt" \
    --root-dir "/data/local/sdk-img/20210812-2/" \
    --num-classes 38 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "ckpt-yylive/scene_cls-5b-2/ckpt/checkpoint-iter-064000.pyth" \
    --out-per-n 10000 \
    --out "sdk-img-20210812-2-scene_cls-5b-2-64k"

##################################################################################################################
# 游戏模型
#
# 8类，非游戏，抽奖，棋类，扑克，麻将，体育类，三国杀/英雄杀，其他游戏
horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/game2/tra-game.txt.1" \
--root-dir "/data/local/" \
--batch-size 64 \
--num-classes 8 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene-game-1" \
--ckpt-save-interval 1000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/game2/tra-game.txt.1" \
--root-dir "/data/local/" \
--batch-size 128 \
--num-classes 8 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 64000,96000,128001 \
--sam 1 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnet18" \
--ckpt-log-dir "ckpt-yylive/scene-game-2" \
--ckpt-save-interval 1000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/game2/tra-game.txt.1" \
--root-dir "/data/local/" \
--batch-size 128 \
--num-classes 8 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64001 \
--sam 1 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnet18" \
--pretrained-ckpt "ckpt-yylive/scene-game-2/ckpt/checkpoint-iter-064000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene-game-2b" \
--ckpt-save-interval 1000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/game2/tra-game.txt.1" \
--root-dir "/data/local/" \
--batch-size 64 \
--num-classes 8 \
--img-size 256 \
--fixres 1 \
--base-lr 0.001 \
--lr-stages-step 500,1001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/scene-game-1/ckpt/checkpoint-iter-005000.pyth" \
--ckpt-log-dir "ckpt-yylive/scene-game-1-fixres" \
--ckpt-save-interval 500 \
--fp16 0


DIRNAME=ckpt-yylive/scene-game-1
Nckpts=16000
Ninterval=1000
for ((i=0; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../yylive/game2/val-game.txt" \
    --root-dir "/data/local/" \
    --num-classes 8 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    --out-per-n 10000 \
    --out "$DIRNAME/val/val-$part_n"
    python hago/val_hago.py \
    "../yylive/game2/val-game.txt" \
    "$DIRNAME/val/val-$part_n" \
    "${DIRNAME}/log/val/" \
    ${i}
done

DIRNAME=ckpt-yylive/scene-game-2b
Nckpts=64000
Ninterval=1000
for ((i=41000; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../yylive/game2/val-game.txt" \
    --root-dir "/data/local/" \
    --num-classes 8 \
    --net "resnet18" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    --out-per-n 10000 \
    --out "$DIRNAME/val/val-$part_n"
    python hago/val_hago.py \
    "../yylive/game2/val-game.txt" \
    "$DIRNAME/val/val-$part_n" \
    "${DIRNAME}/log/val/" \
    ${i}
done


horovodrun -np 8 python hago/predict_hago.py \
    --list-file "/data/local/sdk-img/20210810.txt" \
    --root-dir "/data/local/sdk-img/20210810/" \
    --num-classes 8 \
    --net "resnet18" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "ckpt-yylive/scene-game-2/ckpt/checkpoint-iter-102000.pyth" \
    --out-per-n 10000 \
    --out aaa













horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/search-same/uid-train-gt-10-day-20210901-0911.txt.1" \
--root-dir  "/data/local/search-same/" \
--batch-size 64 \
--num-classes 27589 \
--img-size 224 \
--base-lr 0.1 \
--lr-stages-step 64000,96000,128001 \
--sam 0 \
--final-drop 0.2 \
--reduce-dim 512 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_cls-512-1" \
--ckpt-save-interval 4000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/search-same/uid-train-gt-10-day-20210901-0915.txt.1" \
--root-dir  "/data/local/search-same/" \
--batch-size 64 \
--num-classes 29667 \
--img-size 224 \
--base-lr 0.1 \
--lr-stages-step 128000,192000,256001 \
--sam 1 \
--final-drop 0.2 \
--reduce-dim 512 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_cls-512-2" \
--ckpt-save-interval 4000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/search-same/uid-train-gt-10-day-20210901-1022.txt.1" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/img-sdk/" \
--batch-size 64 \
--num-classes 42430 \
--img-size 224 \
--base-lr 0.1 \
--lr-stages-step 128000,192000,256001 \
--sam 1 \
--final-drop 0.2 \
--reduce-dim 512 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_cls-512-3" \
--ckpt-save-interval 4000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/search-same/uid-train-gt-10-day-20210901-1022.txt.1" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/img-sdk/" \
--batch-size 64 \
--num-classes 42430 \
--img-size 224 \
--base-lr 0.1 \
--lr-stages-step 256000,384000,512001 \
--sam 1 \
--final-drop 0.2 \
--reduce-dim 512 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_cls-512-3l" \
--ckpt-save-interval 4000 \
--fp16 0

