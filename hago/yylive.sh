
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

PREDICT_npy() {
cd /data/remote/Moco/
horovodrun -np 8 python hago/predict_hago.py \
--list-file "$3" \
--root-dir "$4" \
--load-npy 1 \
--num-classes $2 \
--net "resnest50" \
--img-size 256 \
--batch-size 128 \
--ckpt "$1" \
--out-per-n 100000 \
--out "$5"
cd -
}

# FNAME="/data/remote/yylive/uid_has_voice/info/mp4-10min-frames-jpg-0.4fps-20210301.txt"
# FDIR="/data/remote/yylive/uid_has_voice/data3/20210301/"

FNAME="../yylive/multi_task/imgs-list-20210321.txt"
FDIR="/data/local/yylive_video_data/"

PREDICT "ckpt-yylive/proc_play-1c/ckpt/checkpoint-iter-008000.pyth" 5 \
        "$FNAME" "$FDIR" "$FNAME-proc_play"
PREDICT "ckpt-yylive/yybear-1/ckpt/checkpoint-iter-002000.pyth" 2 \
        "$FNAME" "$FDIR" "$FNAME-yybear"
PREDICT "ckpt-yylive/game-2/ckpt/checkpoint-iter-010000.pyth" 2 \
        "$FNAME" "$FDIR" "$FNAME-game"
PREDICT "ckpt-yylive/uid_has-2-fixres/ckpt/checkpoint-iter-001000.pyth" 4 \
        "$FNAME" "$FDIR" "$FNAME-uid_has"
PREDICT "ckpt-yylive/guaping-1/ckpt/checkpoint-iter-006000.pyth" 2 \
        "$FNAME" "$FDIR" "$FNAME-guaping"
# PREDICT "ckpt-yylive/movie-1/ckpt/checkpoint-iter-032000.pyth" 5 \
#         "$FNAME" "$FDIR" "$FNAME-movie"
PREDICT "ckpt-yylive/movie-2/ckpt/checkpoint-iter-040000.pyth" 5 \
        "$FNAME" "$FDIR" "$FNAME-movie"

FNAME="/data/remote/yylive/uid_has_voice/info/mp4-60s-logmel-npy-5s-8-20210123-0306.txt"
FDIR="/data/remote/yylive/uid_has_voice/data2/"
PREDICT_npy "ckpt-yylive/uid_has_voice-5e/ckpt/checkpoint-iter-040000.pyth" 3 \
        "$FNAME" "$FDIR" "$FNAME-uid_has_voice-5e-40k"
FNAME="/data/remote/yylive/uid_has_voice/info/mp4-10min-logmel-npy-5s-0.2fps-20210301.txt"
FDIR="/data/remote/yylive/uid_has_voice/data2/20210301/"
PREDICT_npy "ckpt-yylive/uid_has_voice-4b/ckpt/checkpoint-iter-045000.pyth" 3 \
        "$FNAME" "$FDIR" "$FNAME-uid_has_voice"

FNAME="/data/remote/yylive/uid_has_voice/train-soft-300w.txt"
FDIR="/data/remote/yylive/uid_has_voice/data2/"
PREDICT_npy "ckpt-yylive/uid_has_voice-6-fixres/ckpt/checkpoint-iter-010000.pyth" 3 \
        "$FNAME" "$FDIR" "$FNAME-uid_has_voice"

PREDICT "ckpt-yylive/movie-2/ckpt/checkpoint-iter-040000.pyth" 5 \
        "../yylive/proc_play/info/frames-20210316-proc_play.txt" \
        "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210316/" \
        "movie-2-20210316"
PREDICT "ckpt-yylive/movie-2/ckpt/checkpoint-iter-020000.pyth" 5 \
        "../yylive/proc_play/info/frames-20210316-proc_play.txt" \
        "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210316/" \
        "movie-2b-20210316"



##################################################################################################################
# 测试
FNAME="/data/remote/yylive/testset/tag-voice.txt"
FDIR="http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/app/testset-20210315/logmel/"
PREDICT_npy "ckpt-yylive/uid_has_voice-6-fixres/ckpt/checkpoint-iter-010000.pyth" 3 \
        "$FNAME" "$FDIR" "$FNAME-uid_has_voice"

#FDIR="/data/local/testset/vid-frame/"
FDIR="http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/app/testset-20210315/vid-frame/"
FNAME="/data/remote/yylive/testset/tag/tag-uid_has.txt"
PREDICT "ckpt-yylive/uid_has-2-fixres/ckpt/checkpoint-iter-001000.pyth" 4 \
        "$FNAME" "$FDIR" "$FNAME-uid_has"
PREDICT "ckpt-yylive/uid_has-2-pos-fixres/ckpt/checkpoint-iter-002000.pyth" 4 \
        "$FNAME" "$FDIR" "$FNAME-uid_has2"
FNAME="/data/remote/yylive/testset/tag/tag-guaping.txt"
PREDICT "ckpt-yylive/guaping-1/ckpt/checkpoint-iter-006000.pyth" 2 \
        "$FNAME" "$FDIR" "$FNAME-guaping"
FNAME="/data/remote/yylive/testset/tag/tag-movie.txt"
PREDICT "ckpt-yylive/movie-2/ckpt/checkpoint-iter-040000.pyth" 5 \
        "$FNAME" "$FDIR" "$FNAME-movie"

for FNAME in uid_has guaping movie ; do
FNAME="/data/remote/yylive/testset/tag/tag-$FNAME.txt"
PREDICT "ckpt-yylive/multi_label-1c-sam/ckpt/checkpoint-iter-128000.pyth" 15 \
        "$FNAME" "$FDIR" "$FNAME-mt2"
PREDICT "ckpt-yylive/multi_label-1c-fixres/ckpt/checkpoint-iter-008000.pyth" 15 \
        "$FNAME" "$FDIR" "$FNAME-mtf"
done

# 挂机
FDIR="/data/local/testset/vid-10m-frames/"
FNAME="/data/remote/yylive/testset/mp4-10m-list-1.txt.frames"
PREDICT "ckpt-yylive/multi_label-1c-fixres/ckpt/checkpoint-iter-008000.pyth" 15 \
        "$FNAME" "$FDIR" "$FNAME-mtf"
FDIR="/data/local/testset/vid-10m-logmel/"
FNAME="/data/remote/yylive/testset/mp4-10m-list-1.txt.logmel"
PREDICT_npy "ckpt-yylive/uid_has_voice-6-fixres/ckpt/checkpoint-iter-010000.pyth" 3 \
        "$FNAME" "$FDIR" "$FNAME-uid_has_voice"



##################################################################################################################
# 多标签
horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/multi_task/train-20210321.txt.1" \
--root-dir "/data/local/yylive_video_data/" \
--batch-size 128 \
--soft-label 1 \
--num-classes 4,2,5,2,2 \
--img-size 224 \
--base-lr 0.03 \
--lr-stages-step 64000,96000,128001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/multi_label-1c-sam" \
--ckpt-save-interval 4000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/multi_task/train-20210321.txt.1" \
--root-dir "/data/local/yylive_video_data/" \
--batch-size 128 \
--soft-label 1 \
--num-classes 4,2,5,2,2 \
--img-size 256 \
--base-lr 0.01 \
--fixres 1 \
--lr-stages-step 4000,6000,8001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/multi_label-1c-sam/ckpt/checkpoint-iter-128000.pyth" \
--ckpt-log-dir "ckpt-yylive/multi_label-1c-sam-fixres" \
--ckpt-save-interval 1000 \
--fp16 0


##################################################################################################################
# 挂机，0非，1是
horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/guaji/train-20210220.txt.1" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/" \
--batch-size 128 \
--num-classes 2 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-moco/resnest50-528c19ca.pth" \
--ckpt-log-dir "ckpt-yylive/guaji-1" \
--ckpt-save-interval 1000 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/guaji/train-20210224.txt.1" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/" \
--batch-size 64 \
--num-classes 2 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/guaji-2" \
--ckpt-save-interval 1000 \
--fp16 0

horovodrun -np 8 python hago/predict_hago.py \
--list-file "/data/remote/yylive/proc_play/info/frames-20210228-proc_play.txt" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210228/" \
--num-classes 2 \
--net "resnest50" \
--img-size 256 \
--batch-size 128 \
--ckpt "ckpt-yylive/guaji-2/ckpt/checkpoint-iter-016000.pyth" \
--out-per-n 1000 \
--out aaa-28

horovodrun -np 8 python hago/predict_hago.py \
--list-file "/data/remote/yylive/proc_play/info/frames-20210228-proc_play.txt" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210228/" \
--num-classes 2 \
--net "resnest50" \
--img-size 256 \
--batch-size 128 \
--ckpt "ckpt-yylive/guaji-1/ckpt/checkpoint-iter-016000.pyth" \
--out-per-n 1000 \
--out aaa-28



##################################################################################################################
# 挂屏，0非，1是
horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/guaping/guaping-train-10w.txt.1" \
--root-dir "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210224/" \
--batch-size 64 \
--num-classes 2 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 4000,6000,8001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/guaping-1" \
--ckpt-save-interval 1000 \
--fp16 0

DIRNAME=ckpt-yylive/guaping-1
horovodrun -np 8 python hago/predict_hago.py \
--list-file "/data/remote/yylive/proc_play/info/frames-20210228-proc_play.txt" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210228/" \
--num-classes 2 \
--net "resnest50" \
--img-size 256 \
--batch-size 128 \
--ckpt "$DIRNAME/ckpt/checkpoint-iter-004000.pyth" \
--out-per-n 1000 \
--out aaa-28




##################################################################################################################
# 真人，0非，1主播，2主播小窗，[3非主播]
horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has/uid_has-train-1w.txt.1" \
--root-dir "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210224/" \
--batch-size 64 \
--num-classes 3 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 4000,6000,8001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1c/ckpt/checkpoint-iter-045000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_has-1-uid_cls-init" \
--ckpt-save-interval 500 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has/uid_has-train-1w.txt.1" \
--root-dir "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210224/" \
--batch-size 64 \
--num-classes 3 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 4000,6000,8001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-moco/resnest50-528c19ca.pth" \
--ckpt-log-dir "ckpt-yylive/uid_has-1-imgnet-init" \
--ckpt-save-interval 500 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has/uid_has-train-10w.txt.1" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210224/" \
--batch-size 64 \
--num-classes 4 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_has-2" \
--ckpt-save-interval 1000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has/uid_has-train-10w.txt.1" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210224/" \
--batch-size 64 \
--num-classes 4 \
--img-size 256 \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 2000,3000,4001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_has-2/ckpt/checkpoint-iter-014000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_has-2-fixres" \
--ckpt-save-interval 1000 \
--fp16 0

# 对小窗的图片做特殊增强，不要把人crop掉
horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has/uid_has-train-10w.txt.fine.1" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210224/" \
--batch-size 64 \
--num-classes 4 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 4000,6000,8001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_has-2-pos" \
--ckpt-save-interval 1000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has/uid_has-train-10w.txt.fine.1" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210224/" \
--batch-size 64 \
--num-classes 4 \
--img-size 256 \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 1000,1500,2001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_has-2-pos/ckpt/checkpoint-iter-008000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_has-2-pos-fixres" \
--ckpt-save-interval 500 \
--fp16 0


# 合并 非主播 和 无人
horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has/uid_has-train-10w-b.txt.1" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210224/" \
--batch-size 64 \
--num-classes 4 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_has-2b" \
--ckpt-save-interval 1000 \
--fp16 0

DIRNAME=ckpt-yylive/uid_has-2
horovodrun -np 8 python hago/predict_hago.py \
--list-file "/data/remote/yylive/proc_play/info/frames-20210228-proc_play.txt" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210228/" \
--num-classes 4 \
--net "resnest50" \
--img-size 256 \
--batch-size 128 \
--ckpt "$DIRNAME/ckpt/checkpoint-iter-010000.pyth" \
--out-per-n 1000 \
--out aaa-28

DIRNAME=ckpt-yylive/uid_has-2-fixres
horovodrun -np 8 python hago/predict_hago.py \
--list-file "/data/remote/yylive/proc_play/info/frames-20210228-proc_play.txt" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210228/" \
--num-classes 4 \
--net "resnest50" \
--img-size 256 \
--batch-size 128 \
--ckpt "$DIRNAME/ckpt/checkpoint-iter-001000.pyth" \
--out-per-n 1000 \
--out aaa-28


DIRNAME=ckpt-yylive/uid_has-1-uid_cls-init
DIRNAME=ckpt-yylive/uid_has-1-imgnet-init
for part_n in 000000 000500 001000 001500 002000 002201 002701 003201 003701 004201 004701 \
                005201 005701 006201 006701 007201 007701 008001; do
#for part_n in 000000 000500 001000 001500 002000 002200 002699 003199 003699 004199 004699 \
#                005199 005699 006199 006699 007199 007699 008001; do
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "/data/remote/yylive/uid_has/uid_has-val-2k.txt" \
    --root-dir  "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210224/" \
    --num-classes 3 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 128 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    --out-per-n 1000 \
    --out  "$DIRNAME/val/uid_has-clsinit-${part_n}"
done

for part_n in 000000 000500 001000 001500 002000 002201 002701 003201 003701 004201 004701 \
                005201 005701 006201 006701 007201 007701 008001; do
# for part_n in 000000 000500 001000 001500 002000 002200 002699 003199 003699 004199 004699 \
#                 005199 005699 006199 006699 007199 007699 008001; do
python hago/val_hago.py \
    "/data/remote/yylive/uid_has/uid_has-val-2k.txt" \
    "$DIRNAME/val/uid_has-clsinit-${part_n}" \
    "${DIRNAME}/log/val/" \
    ${part_n}
done

DIRNAME=ckpt-yylive/uid_has-1-imgnet-init
DIRNAME=ckpt-yylive/uid_has-1-uid_cls-init
part_n=002000
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../yylive/proc_play/info/frames-20210227-proc_play.txt" \
    --root-dir  "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210227/" \
    --num-classes 3 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 128 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    --out-per-n 1000 \
    --out  bbbb1

horovodrun -np 8 python hago/predict_hago.py \
--list-file "../yylive/uid_has_voice/mp4-4s-jpg-20210123-0306.txt" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/mp4-frames-2s/" \
--num-classes 4 \
--net "resnest50" \
--img-size 256 \
--batch-size 64 \
--ckpt "ckpt-yylive/uid_has-2-fixres/ckpt/checkpoint-iter-001000.pyth" \
--out-per-n 10000 \
--out ../yylive/uid_has_voice/mp4-4s-jpg-20210123-0306.txt--uid_has-2-fixres-1k



##################################################################################################################
# 人声，0非，1讲话，2唱歌
for i in '29-224-256' '29-160-184' '29_thin-160-184' '17-160-184' '17-112-128' '17_thin-112-128'; do
    NET=`echo $i|awk -F"-" '{print "resnest"$1}'`
    SIZE=`echo $i|awk -F"-" '{print $2}'`
    horovodrun -np 8 python hago/train_hago.py \
        --list-file "../yylive/uid_has_voice/uid_cls-train-v3-gt30-15757.txt.1" \
        --root-dir "/data/local/uid_has_voice2/" \
        --load-npy 1 \
        --batch-size 64 \
        --num-classes 15757 \
        --img-size $SIZE \
        --base-lr 0.03 \
        --lr-stages-step 128000,192000,256001 \
        --mixup 0.2 \
        --sam 0 \
        --final-drop 0.2 \
        --no-bias-bn-wd 1 \
        --weight-decay 1e-4 \
        --net "$NET" \
        --ckpt-log-dir "ckpt-yylive/uid_has_voice-uid_cls-4-resnest$i" \
        --ckpt-save-interval 8000 \
        --fp16 0
    SIZE=`echo $i|awk -F"-" '{print $3}'`
    horovodrun -np 8 python hago/train_hago.py \
        --list-file "../yylive/uid_has_voice/uid_cls-train-v3-gt30-15757.txt.1" \
        --root-dir "/data/local/uid_has_voice2/" \
        --load-npy 1 \
        --batch-size 64 \
        --num-classes 15757 \
        --img-size $SIZE \
        --fixres 1 \
        --base-lr 0.01 \
        --lr-stages-step 32000,48000,64001 \
        --mixup 0 \
        --sam 1 \
        --final-drop 0.2 \
        --no-bias-bn-wd 1 \
        --weight-decay 1e-4 \
        --net "$NET" \
        --pretrained-ckpt "ckpt-yylive/uid_has_voice-uid_cls-4-resnest$i/ckpt/checkpoint-iter-256000.pyth" \
        --ckpt-log-dir "ckpt-yylive/uid_has_voice-uid_cls-4-resnest$i-fixres" \
        --ckpt-save-interval 4000 \
        --fp16 0
done

for ii in '29-224-256' '29-160-184' '29_thin-160-184' '17-160-184' '17-112-128' '17_thin-112-128'; do
    NET=`echo $ii|awk -F"-" '{print "resnest"$1}'`
    SIZE=`echo $ii|awk -F"-" '{print $3}'`
    DIRNAME=ckpt-yylive/uid_has_voice-uid_cls-4-resnest$ii
    Nckpts=256000
    Ninterval=8000
    for ((i=$Nckpts; i<=$Nckpts; i+=$Ninterval)); do
        part_n=`printf "%06d" ${i}`
        horovodrun -np 8 python hago/predict_hago.py \
        --list-file "../yylive/uid_has_voice/uid_cls-val-v3-gt30-15757.txt" \
        --root-dir "/data/local/uid_has_voice/train/" \
        --load-npy 1 \
        --num-classes 15757 \
        --net "$NET" \
        --img-size $SIZE \
        --batch-size 32 \
        --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
        --out-per-n 1000 \
        --out "$DIRNAME/val/val-$part_n"
        python hago/val_hago.py \
        "../yylive/uid_has_voice/uid_cls-val-v3-gt30-15757.txt" \
        "$DIRNAME/val/val-$part_n" \
        "${DIRNAME}/log/val/" \
        ${i}
    done
    DIRNAME=ckpt-yylive/uid_has_voice-uid_cls-4-resnest$ii-fixres
    Nckpts=64000
    Ninterval=4000
    for ((i=$Nckpts; i<=$Nckpts; i+=$Ninterval)); do
        part_n=`printf "%06d" ${i}`
        horovodrun -np 8 python hago/predict_hago.py \
        --list-file "../yylive/uid_has_voice/uid_cls-val-v3-gt30-15757.txt" \
        --root-dir "/data/local/uid_has_voice/train/" \
        --load-npy 1 \
        --num-classes 15757 \
        --net "$NET" \
        --img-size $SIZE \
        --batch-size 32 \
        --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
        --out-per-n 1000 \
        --out "$DIRNAME/val/val-$part_n"
        python hago/val_hago.py \
        "../yylive/uid_has_voice/uid_cls-val-v3-gt30-15757.txt" \
        "$DIRNAME/val/val-$part_n" \
        "${DIRNAME}/log/val/" \
        ${i}
    done
done


for i in '29-224-256' '29-160-184' '29_thin-160-184' '17-160-184' '17-112-128' '17_thin-112-128'; do
    NET=`echo $i|awk -F"-" '{print "resnest"$1}'`
    SIZE=`echo $i|awk -F"-" '{print $2}'`
    horovodrun -np 8 python hago/train_hago.py \
        --list-file "../yylive/uid_has_voice/train.txt.5" \
        --root-dir "/data/local/uid_has_voice/train/" \
        --load-npy 1 \
        --batch-size 64 \
        --num-classes 3 \
        --img-size $SIZE \
        --base-lr 0.01 \
        --lr-stages-step 16000,24000,32001 \
        --mixup 1.0 \
        --sam 1 \
        --final-drop 0.2 \
        --no-bias-bn-wd 1 \
        --weight-decay 1e-4 \
        --net "$NET" \
        --pretrained-ckpt "ckpt-yylive/uid_has_voice-uid_cls-4-resnest$i-fixres/ckpt/checkpoint-iter-064000.pyth" \
        --ckpt-log-dir "ckpt-yylive/uid_has_voice-5-ft-$i" \
        --ckpt-save-interval 1000 \
        --fp16 0
done

for ii in '29-224-256' '29-160-184' '29_thin-160-184' '17-160-184' '17-112-128' '17_thin-112-128'; do
    NET=`echo $ii|awk -F"-" '{print "resnest"$1}'`
    SIZE=`echo $ii|awk -F"-" '{print $3}'`
    DIRNAME=ckpt-yylive/uid_has_voice-5-ft-$ii
    Nckpts=32000
    Ninterval=1000
for ((i=$Nckpts; i<=$Nckpts; i+=$Ninterval)); do
part_n=`printf "%06d" ${i}`
echo $i, 'wait'; while [ 1 ] ; do
    [[ -f "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" ]] && break
    sleep 1s
done; sleep 1s; echo $i, 'begin'
horovodrun -np 8 python hago/predict_hago.py \
--list-file "../yylive/uid_has_voice/val.txt.v3" \
--root-dir "/data/local/uid_has_voice/train/" \
--load-npy 1 \
--num-classes 3 \
--net "$NET" \
--img-size $SIZE \
--batch-size 32 \
--ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
--out-per-n 1000 \
--out "$DIRNAME/val/val-$part_n"
python hago/val_hago.py \
"../yylive/uid_has_voice/val.txt.v3" \
"$DIRNAME/val/val-$part_n" \
"${DIRNAME}/log/val/" \
${i}
done
done

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has_voice/uid_cls-train-v3-gt30-15757.txt.1" \
--root-dir "/data/local/uid_has_voice2/" \
--load-npy 1 \
--batch-size 64 \
--num-classes 15757 \
--img-size 224 \
--base-lr 0.03 \
--lr-stages-step 128000,160000,192001 \
--mixup 0.2 \
--sam 0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest101" \
--pretrained-ckpt "ckpt-moco/resnest101-22405ba7.pth" \
--ckpt-log-dir "ckpt-yylive/uid_has_voice-uid_cls-4-resnest101-2" \
--ckpt-save-interval 4000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has_voice/uid_cls-train-v3-gt30-15757.txt.1" \
--root-dir "/data/local/uid_has_voice2/" \
--load-npy 1 \
--batch-size 64 \
--num-classes 15757 \
--img-size 256 \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 32000,48000,64001 \
--mixup 0 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest101" \
--pretrained-ckpt "ckpt-yylive/uid_has_voice-uid_cls-4-resnest101-2/ckpt/checkpoint-iter-192000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_has_voice-uid_cls-4-resnest101-2-fixres" \
--ckpt-save-interval 4000 \
--fp16 0

DIRNAME=ckpt-yylive/uid_has_voice-uid_cls-4-resnest101-2-fixres
Nckpts=64000
Ninterval=4000
for ((i=0; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    
    echo $i, 'wait'; while [ 1 ] ; do
        [[ -f "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" ]] && break
        sleep 1s
    done; sleep 1s; echo $i, 'begin'
    
    horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../yylive/uid_has_voice/uid_cls-val-v3-gt30-15757.txt" \
    --root-dir \
    "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/app/uid_has_voice/" \
    --load-npy 1 \
    --num-classes 15757 \
    --net "resnest101" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    --out-per-n 1000 \
    --out "$DIRNAME/val/val-$part_n"
    python hago/val_hago.py \
    "../yylive/uid_has_voice/uid_cls-val-v3-gt30-15757.txt" \
    "$DIRNAME/val/val-$part_n" \
    "${DIRNAME}/log/val/" \
    ${i}
done

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has_voice/train.txt.5" \
--root-dir "/data/local/uid_has_voice/train/" \
--load-npy 1 \
--batch-size 64 \
--num-classes 3 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 16000,20000,24001 \
--mixup 1.0 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest101" \
--pretrained-ckpt "ckpt-yylive/uid_has_voice-uid_cls-4-resnest101-2-fixres/ckpt/checkpoint-iter-064000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_has_voice-5e-3b" \
--ckpt-save-interval 500 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has_voice/train.txt.v5.fg.1" \
--root-dir "/data/local/uid_has_voice/train/" \
--load-npy 1 \
--fine-grained 1 \
--fine-grained-n 5 \
--batch-size 64 \
--num-classes 3 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 32000,36000,40001 \
--mixup 1.0 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_has_voice-uid_cls-4-fixres/ckpt/checkpoint-iter-064000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_has_voice-7b" \
--ckpt-save-interval 1000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has_voice/train-soft-300w.txt.2" \
--root-dir "/data/local/uid_has_voice2/" \
--load-npy 1 \
--soft-label 1 \
--batch-size 64 \
--num-classes 3 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 64000,72000,80001 \
--mixup 0 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_has_voice-uid_cls-4-fixres/ckpt/checkpoint-iter-064000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_has_voice-6b" \
--ckpt-save-interval 1000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_has_voice/train.txt.5" \
--root-dir "/data/local/uid_has_voice/train/" \
--load-npy 1 \
--batch-size 64 \
--num-classes 3 \
--img-size 256 \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 2000,3000,4001 \
--mixup 0 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_has_voice-6b/ckpt/checkpoint-iter-080000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_has_voice-6b-fixres-2" \
--ckpt-save-interval 500 \
--fp16 0

DIRNAME=ckpt-yylive/uid_has_voice-5e-3b
Nckpts=24000
Ninterval=500
for ((i=16000; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    
    echo $i, 'wait'; while [ 1 ] ; do
        [[ -f "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" ]] && break
        sleep 1s
    done; sleep 1s; echo $i, 'begin'
    
    horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../yylive/uid_has_voice/val.txt.v3" \
    --root-dir "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/app/uid_has_voice/" \
    --load-npy 1 \
    --num-classes 3 \
    --net "resnest101" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    --out-per-n 1000 \
    --out "$DIRNAME/val/val-$part_n"
    python hago/val_hago.py \
    "../yylive/uid_has_voice/val.txt.v3" \
    "$DIRNAME/val/val-$part_n" \
    "${DIRNAME}/log/val/" \
    ${i} 3
done

horovodrun -np 8 python hago/predict_hago.py \
--list-file "../yylive/uid_has_voice/info/mp4-mp3-npy-20210306.txt" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/app/mp4-5s-logmel/20210306/" \
--load-npy 1 \
--num-classes 3 \
--net "resnest50" \
--img-size 256 \
--batch-size 64 \
--ckpt "ckpt-yylive/uid_has_voice-4b/ckpt/checkpoint-iter-045000.pyth" \
--out-per-n 10000 \
--out ../yylive/uid_has_voice/mp4-mp3-npy-20210306--4b-45k

##################################################################################################################
# 是否 影视, 0非, 1电影电视剧, 2动画, 3体育, 4综艺
horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/movie/train-20210224.txt.1" \
--root-dir "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/" \
--batch-size 64 \
--num-classes 5 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1c/ckpt/checkpoint-iter-045000.pyth" \
--ckpt-log-dir "ckpt-yylive/movie-1" \
--ckpt-save-interval 2000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/movie/train-20210322.txt.1" \
--root-dir "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/" \
--batch-size 64 \
--num-classes 5 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 24000,32000,40001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/movie-2" \
--ckpt-save-interval 2000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/movie/train-20210322.txt.2" \
--root-dir "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/" \
--batch-size 64 \
--num-classes 5 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 16000,18000,20001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/movie-2b" \
--ckpt-save-interval 2000 \
--fp16 0

DIRNAME=ckpt-yylive/movie-1
horovodrun -np 8 python hago/predict_hago.py \
--list-file "/data/remote/yylive/proc_play/info/frames-20210228-proc_play.txt" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210228/" \
--num-classes 5 \
--net "resnest50" \
--img-size 256 \
--batch-size 128 \
--ckpt "$DIRNAME/ckpt/checkpoint-iter-032000.pyth" \
--out-per-n 1000 \
--out aaa-28

horovodrun -np 8 python hago/predict_hago.py \
--list-file "../yylive/uid_has_voice/mp4-4s-jpg-20210123-0306.txt" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/mp4-frames-2s/" \
--num-classes 5 \
--net "resnest50" \
--img-size 256 \
--batch-size 64 \
--ckpt "ckpt-yylive/movie-1/ckpt/checkpoint-iter-032000.pyth" \
--out-per-n 10000 \
--out ../yylive/uid_has_voice/mp4-4s-jpg-20210123-0306.txt--movie-1-32k



##################################################################################################################
# uid分类，几万类
# *** sam 在前500步关闭 ***
horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_cls/uid-train-28185-gt-10-day-20210201-25.txt.1" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/" \
--batch-size 128 \
--num-classes 28185 \
--img-size 224 \
--base-lr 0.1 \
--lr-stages-step 32000,48000,64001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1/ckpt/checkpoint-iter-001000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_cls-1c" \
--ckpt-save-interval 1000 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_cls/uid-train-34159-gt-10-day-20210123-0227.txt.1" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/" \
--batch-size 64 \
--num-classes 34159 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 64000,128001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1c/ckpt/checkpoint-iter-045000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_cls-1d" \
--ckpt-save-interval 1000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_cls/uid-train-32776-gt-10-day-20210123-0227.txt.rmyybear.1" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/" \
--batch-size 64 \
--num-classes 32777 \
--img-size 224 \
--base-lr 0.1 \
--lr-stages-step 16000,64000,96001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1d/ckpt/checkpoint-iter-065000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_cls-1e" \
--ckpt-save-interval 4000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/uid_cls/uid-train-32776-gt-10-day-20210123-0227.txt.rmyybear.1" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/" \
--batch-size 64 \
--num-classes 32777 \
--img-size 224 \
--base-lr 0.1 \
--lr-stages-step 32000,64000,96001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/uid_cls-1f" \
--ckpt-save-interval 4000 \
--fp16 0


DIRNAME=ckpt-yylive/uid_cls-1e
horovodrun -np 8 python hago/predict_hago.py \
--list-file "/data/remote/yylive/proc_play/info/frames-20210301-proc_play.txt" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210301/" \
--num-classes 0 \
--net "resnest50" \
--img-size 256 \
--batch-size 128 \
--ckpt "$DIRNAME/ckpt/checkpoint-iter-096000.pyth" \
--out-per-n 1000 \
--out uid_cls-0301




##################################################################################################################
# 分屏分类，0非，1两分屏，2六分屏，3四分屏，4六分屏2
DIRNAME=ckpt-yylive/proc_play-1b
DIRNAME=ckpt-yylive/proc_play-1b-fixres-256-layer4
Nckpts=8
Ninterval=1
RES=256

for ((i=0; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%03d" ${i}`
    
    echo $i, 'wait'; while [ 1 ] ; do
        [[ -f "${DIRNAME}/ckpt/checkpoint-iter-${part_n}000.pyth" ]] && break
        sleep 1s
    done; sleep 1s; echo $i, 'begin'
    
    horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../yylive/proc_play/proc_play-val-b.txt" \
    --root-dir  "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210223/" \
    --num-classes 5 \
    --net "resnest50" \
    --img-size $RES \
    --batch-size 128 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}000.pyth" \
    --out-per-n 1000 \
    --out  "$DIRNAME/val/3k-${part_n}000"
    python hago/val_hago.py \
        "../yylive/proc_play/proc_play-val.txt" \
        "$DIRNAME/val/3k-${part_n}000" \
        "${DIRNAME}/log/val/" \
        ${i}000
done


horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/proc_play/proc_play-train.txt.1" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/" \
--batch-size 128 \
--num-classes 3 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16100 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-moco/resnest50-528c19ca.pth" \
--ckpt-log-dir "ckpt-yylive/proc_play-1" \
--ckpt-save-interval 1000 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/proc_play/proc_play-train.txt.1" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/" \
--batch-size 128 \
--num-classes 3 \
--img-size 256 \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 4000,6000,8100 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/proc_play-1/ckpt/checkpoint-iter-016000.pyth" \
--ckpt-log-dir "ckpt-yylive/proc_play-1-fixres-256-layer4" \
--ckpt-save-interval 1000 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/proc_play/proc_play-train.txt.1" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/" \
--batch-size 128 \
--num-classes 5 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 4000,6000,8001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-moco/resnest50-528c19ca.pth" \
--ckpt-log-dir "ckpt-yylive/proc_play-1c" \
--ckpt-save-interval 1000 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/proc_play/proc_play-train.txt.1" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/" \
--batch-size 128 \
--num-classes 5 \
--img-size 256 \
--fixres 1 \
--base-lr 0.01 \
--lr-stages-step 4000,6000,8100 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/proc_play-1b/ckpt/checkpoint-iter-016000.pyth" \
--ckpt-log-dir "ckpt-yylive/proc_play-1b-fixres-256-layer4" \
--ckpt-save-interval 1000 \
--fp16 1



DIRNAME=ckpt-yylive/proc_play-1c
part_n=008
DIRNAME=ckpt-yylive/guaji-1
part_n=016
day=20210224
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../yylive/livedata/info/frames-$day.txt" \
    --root-dir "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/$day/" \
    --num-classes 5 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 128 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}000.pyth" \
    --out-per-n 1000 \
    --out  "proc_play-$day"
DIRNAME=ckpt-yylive/guaji-1
part_n=016
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "/data/remote/yylive/guaji/img-list" \
    --root-dir  "/data/local/guaji/fff/" \
    --num-classes 2 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 128 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}000.pyth" \
    --out-per-n 1000 \
    --out  "bbbb-guaji-mp4-24"




##################################################################################################################
# 是否 yy熊，0非，1是
horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/yybear/yybear-train-20210224-2.4w.txt.1" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210224/" \
--batch-size 64 \
--num-classes 2 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 4000,6000,8001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1c/ckpt/checkpoint-iter-045000.pyth" \
--ckpt-log-dir "ckpt-yylive/yybear-1" \
--ckpt-save-interval 500 \
--fp16 0

DIRNAME=ckpt-yylive/yybear-1
for ((i=0; i<=20; i+=5)); do
part_n=`printf "%04d" ${i}`
horovodrun -np 8 python hago/predict_hago.py \
    --list-file "/data/remote/yylive/yybear/yybear-val.txt" \
    --root-dir \
    "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210224/" \
    --num-classes 2 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 128 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}00.pyth" \
    --out-per-n 1000 \
    --out  "$DIRNAME/val/val-${part_n}00"
python hago/val_hago.py \
    "/data/remote/yylive/yybear/yybear-val.txt" \
    "$DIRNAME/val/val-${part_n}00" \
    "${DIRNAME}/log/val/" \
    ${part_n}00
done

DIRNAME=ckpt-yylive/yybear-1
horovodrun -np 8 python hago/predict_hago.py \
--list-file "/data/remote/yylive/uid_cls/uid-train-34159-gt-10-day-20210123-0227.txt" \
--root-dir "http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/" \
--num-classes 2 \
--net "resnest50" \
--img-size 256 \
--batch-size 128 \
--ckpt "$DIRNAME/ckpt/checkpoint-iter-002000.pyth" \
--out-per-n 1000 \
--out aaa

DIRNAME=ckpt-yylive/yybear-1
horovodrun -np 8 python hago/predict_hago.py \
--list-file "/data/remote/yylive/proc_play/info/frames-20210227-proc_play.txt" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210227/" \
--num-classes 2 \
--net "resnest50" \
--img-size 256 \
--batch-size 128 \
--ckpt "$DIRNAME/ckpt/checkpoint-iter-002000.pyth" \
--out-per-n 1000 \
--out aaa-27




##################################################################################################################
# 是否 游戏, 0非, 1是
# game-1 有 bug，游戏pa的标注处理错误
horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/game/train-20210224.txt.1" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210224/" \
--batch-size 64 \
--num-classes 2 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1c/ckpt/checkpoint-iter-045000.pyth" \
--ckpt-log-dir "ckpt-yylive/game-1" \
--ckpt-save-interval 1000 \
--fp16 0

horovodrun -np 8 python hago/train_hago.py \
--list-file "../yylive/game/train-20210224.txt.1" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210224/" \
--batch-size 64 \
--num-classes 2 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "ckpt-yylive/uid_cls-1e/ckpt/checkpoint-iter-096000.pyth" \
--ckpt-log-dir "ckpt-yylive/game-2" \
--ckpt-save-interval 1000 \
--fp16 0

DIRNAME=ckpt-yylive/game-2
horovodrun -np 8 python hago/predict_hago.py \
--list-file "/data/remote/yylive/proc_play/info/frames-20210228-proc_play.txt" \
--root-dir \
"http://filer.ai.yy.com:9899/dataset/projects/ai/cv/yylive-content-understanding/yylive_video_data/frames/20210228/" \
--num-classes 2 \
--net "resnest50" \
--img-size 256 \
--batch-size 128 \
--ckpt "$DIRNAME/ckpt/checkpoint-iter-009000.pyth" \
--out-per-n 1000 \
--out aaa-28


