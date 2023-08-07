horovodrun -np 8 python /data1/liuyidi/moco/four_pic/train_hago_4_2.py \
--list-file "/data1/liuyidi/scene_cls/V4.1.2/list/train_list/data*4_4_list_new.txt" \
--root-dir "/data1/liuyidi/scene_cls/6b_2/" \
--load-npy 0 \
--batch-size 4 \
--num-classes 43 \
--img-size 384 \
--base-lr 0.00005 \
--sam 0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "setting" \
--pretrained-ckpt '' \
--feature_path '/data1/liuyidi/scene_cls/V4.1.2/log_dir/swin_base_22kft1k_1024k_fix_64k_cpb/ckpt/checkpoint-iter-064000.pyth' \
--ckpt-log-dir "/data1/liuyidi/scene_cls/V4.1.2/log_dir_2/trans_4_64k" \
--ckpt-save-interval 1000 \
--rand-corner 0 \
--fp16 0 \
--optimizer "adamw" \
--warmup_step 4000 \
--cos_step 64000 \

--only_trans 1 \
--fixres_2 0 


DIRNAME=/data1/liuyidi/scene_cls/V4.1.2/log_dir_2/trans_4_64k
Nckpts=43000
Ninterval=1000
for ((i= 43000; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    horovodrun -np 8 /data1/liuyidi/moco/four_pic/predict_hago_4_linear2.py \
    --list-file "/data1/liuyidi/scene_cls/V4.1.2/list/train_list/val*4_4_list_new.txt" \
    --root-dir "/data1/liuyidi/scene_cls/6b_2/" \
    --load-npy 0 \
    --num-classes 43 \
    --net "setting" \
    --img-size 384 \
    --batch-size 8 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    --out-per-n 10000 \
    --val_4 1 \
    --out "$DIRNAME/val_clsfix/val-$part_n"
    python /data1/liuyidi/moco/hago/val_hago_*4.py \
    "/data1/liuyidi/scene_cls/V4.1.2/list/train_list/val*4_4_list_new.txt" \
    "$DIRNAME/val_clsfix/val-$part_n" \
    "${DIRNAME}/log/val_clsfix/" \
    ${i} &
done