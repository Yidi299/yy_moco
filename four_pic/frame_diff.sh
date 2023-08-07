horovodrun -np 8 python /data1/liuyidi/moco/four_pic/train_hago_diff.py \
--list-file "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/data*4_4_list_new_resample.txt" \
--root-dir "/data1/liuyidi/scene_cls/6b_2/" \
--load-npy 0 \
--batch-size 8 \
--num-classes 43 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 128000,192000,256001 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt "/data1/liuyidi/scene_cls/weight/uid_cls-1e-checkpoint-iter-096000.pyth" \
--ckpt-log-dir "/home/pengguozhu/liuyidi/logdir/V4.1_4/diff_try0" \
--ckpt-save-interval 4000 \
--rand-corner 1 \
--fp16 0

DIRNAME=/home/pengguozhu/liuyidi/logdir/V4.1_4/diff_try0
Nckpts=256000
Ninterval=4000
for ((i=88000; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    horovodrun -np 8 /data1/liuyidi/moco/four_pic/predict_hago_diff_2.py \
    --list-file "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/val*4_list.txt" \
    --root-dir "/data1/liuyidi/scene_cls/6b_2/" \
    --load-npy 0 \
    --num-classes 43 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    --out-per-n 10000 \
    --val_4 1 \
    --out "$DIRNAME/val_clsfix/val-$part_n"
    python /data1/liuyidi/moco/hago/val_hago_*4.py \
    "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/val*4_list.txt" \
    "$DIRNAME/val_clsfix/val-$part_n" \
    "${DIRNAME}/log/val_clsfix/" \
    ${i} &
done