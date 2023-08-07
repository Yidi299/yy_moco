horovodrun -np 8 python /data1/liuyidi/moco/four_pic/train_hago_4_2.py \
--list-file "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/data*4_4_list.txt" \
--root-dir "/data1/liuyidi/scene_cls/6b_2/" \
--load-npy 0 \
--batch-size 64 \
--num-classes 43 \
--img-size 224 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16000 \
--sam 0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--ckpt-log-dir "/data1/liuyidi/scene_cls/V4.1/log_dir/V4.2_lstm_xavier" \
--ckpt-save-interval 1000 \
--rand-corner 0 \
--fp16 0



        raise HorovodInternalError(e)


# --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \


DIRNAME=/data1/liuyidi/scene_cls/V4.1/log_dir/V4.2_lstm_xavier
Nckpts=16000
Ninterval=1000
for ((i=16000; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    # horovodrun -np 7 /data1/liuyidi/moco/four_pic/predict_hago_4_linear2.py \
    # --list-file "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/val*4_list.txt" \
    # --root-dir "/data1/liuyidi/scene_cls/6b_2/" \
    # --load-npy 0 \
    # --num-classes 43 \
    # --net "resnest50" \
    # --img-size 256 \
    # --batch-size 64 \
    # --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    # --out-per-n 10000 \
    # --val_4 1 \
    # --out "$DIRNAME/val_clsfix/val-$part_n"
    python /data1/liuyidi/moco/hago/val_hago_*4.py \
    "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/val*4_list.txt" \
    "$DIRNAME/val_clsfix/val-$part_n" \
    "${DIRNAME}/log/val_clsfix/" \
    ${i} &
done

horovodrun -np 8 python /data1/liuyidi/moco/four_pic/train_hago_4_2.py \
--list-file "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/data*4_4_list_new.txt" \
--root-dir "/data1/liuyidi/scene_cls/6b_2/" \
--load-npy 0 \
--batch-size 64 \
--num-classes 43 \
--img-size 224 \
--base-lr 0.00005 \
--lr-stages-step 16000,24000,32000 \
--sam 0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--ckpt-log-dir "/data1/liuyidi/scene_cls/V4.1/log_dir_new/V4.2_trans_64k_0.00005_2" \
--ckpt-save-interval 1000 \
--rand-corner 1 \
--fp16 0 \
--optimizer "adamw" \
--warmup_step 6000 \
--cos_step 64000

sleep 5h
DIRNAME=/data1/liuyidi/scene_cls/V4.1/log_dir_new/V4.2_trans_64k_0.00005_2
Nckpts=69000
Ninterval=1000
for ((i=0; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    horovodrun -np 8 /data1/liuyidi/moco/four_pic/predict_hago_4_linear2.py \
    --list-file "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/val*4_list.txt" \
    --root-dir "/data1/liuyidi/scene_cls/6b_2/" \
    --load-npy 0 \
    --num-classes 43 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 64 \
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






DIRNAME=/data1/liuyidi/scene_cls/V4.1/log_dir/V4.2_duibi4
Nckpts=8000
Ninterval=1000
for ((i=8000; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    # horovodrun -np 7 /data1/liuyidi/moco/four_pic/predict_hago_4_linear2.py \
    # --list-file "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/val*4_list.txt" \
    # --root-dir "/data1/liuyidi/scene_cls/6b_2/" \
    # --load-npy 0 \
    # --num-classes 43 \
    # --net "resnest50" \
    # --img-size 256 \
    # --batch-size 64 \
    # --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    # --out-per-n 10000 \
    # --val_4 1 \
    # --out "$DIRNAME/val_clsfix/val-$part_n"
    python /data1/liuyidi/moco/hago/val_hago_*4.py \
    "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/val*4_list.txt" \
    "$DIRNAME/val_clsfix/val-$part_n" \
    "${DIRNAME}/log/val_clsfix/" \
    ${i} &
done



horovodrun -np 8 python /data1/liuyidi/moco/four_pic/train_hago_4_2.py \
--list-file "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/data*4_4_list_new.txt" \
--root-dir "/data1/liuyidi/scene_cls/6b_2/" \
--load-npy 0 \
--batch-size 8 \
--num-classes 43 \
--img-size 224 \
--base-lr 0.00005 \
--lr-stages-step 128000,192000,256000 \
--sam 0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--ckpt-log-dir "/home/pengguozhu/liuyidi/logdir/V4.1_4/V4.2_trans_256k_all_drop0.3_two" \
--ckpt-save-interval 4000 \
--rand-corner 1 \
--fp16 0 \
--optimizer "two" \
--warmup_step 24000 \
--cos_step 256000

DIRNAME=/home/pengguozhu/liuyidi/logdir/V4.1_4/V4.2_trans_256k_all_drop0.3_two
Nckpts=252000
Ninterval=4000
for ((i=252000; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    # horovodrun -np 8 /data1/liuyidi/moco/four_pic/predict_hago_4_linear2.py \
    # --list-file "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/val*4_list.txt" \
    # --root-dir "/data1/liuyidi/scene_cls/6b_2/" \
    # --load-npy 0 \
    # --num-classes 43 \
    # --net "resnest50" \
    # --img-size 256 \
    # --batch-size 64 \
    # --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    # --out-per-n 10000 \
    # --val_4 1 \
    # --out "$DIRNAME/val_clsfix/val-$part_n"
    python /data1/liuyidi/moco/hago/val_hago_*4.py \
    "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/val*4_list.txt" \
    "$DIRNAME/val_clsfix/val-$part_n" \
    "${DIRNAME}/log/val_clsfix/" \
    ${i} &
done


horovodrun -np 8 python /data1/liuyidi/moco/four_pic/train_hago_4_2.py \
--list-file "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/data*4_4_list_new.txt" \
--root-dir "/data1/liuyidi/scene_cls/6b_2/" \
--load-npy 0 \
--batch-size 64 \
--num-classes 43 \
--img-size 224 \
--base-lr 0.00005 \
--lr-stages-step 16000,24000,32000 \
--sam 0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt '/home/pengguozhu/liuyidi/logdir/V4.1_4/V4.2_trans_256k_all_drop0.3_two/ckpt/checkpoint-iter-252000.pyth' \
--ckpt-log-dir "/home/pengguozhu/liuyidi/logdir/V4.1_4/V4.2_trans_256k_all_drop0.3_two_trans_64k" \
--ckpt-save-interval 1000 \
--rand-corner 1 \
--fp16 0 \
--optimizer "adamw" \
--warmup_step 6000 \
--cos_step 64000 \
--only_trans 1
sleep 10s
DIRNAME=/home/pengguozhu/liuyidi/logdir/V4.1_4/V4.2_trans_256k_all_drop0.3_two_trans_64k
Nckpts=44000
Ninterval=1000
for ((i=44000; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    # horovodrun -np 8 /data1/liuyidi/moco/four_pic/predict_hago_4_linear2.py \
    # --list-file "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/val*4_list.txt" \
    # --root-dir "/data1/liuyidi/scene_cls/6b_2/" \
    # --load-npy 0 \
    # --num-classes 43 \
    # --net "resnest50" \
    # --img-size 256 \
    # --batch-size 64 \
    # --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    # --out-per-n 10000 \
    # --val_4 1 \
    # --out "$DIRNAME/val_clsfix/val-$part_n"
    python /data1/liuyidi/moco/hago/val_hago_yiji.py \
    "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/val*4_list.txt" \
    "$DIRNAME/val_clsfix/val-$part_n" \
    "${DIRNAME}/log/val_clsfix/" \
    ${i} &
done

horovodrun -np 8 python /data1/liuyidi/moco/four_pic/train_hago_4_2.py \
--list-file "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/data*4_4_list_new.txt" \
--root-dir "/data1/liuyidi/scene_cls/6b_2/" \
--load-npy 0 \
--batch-size 64 \
--num-classes 43 \
--img-size 224 \
--base-lr 0.00005 \
--lr-stages-step 16000,24000,32000 \
--sam 0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--net "resnest50" \
--pretrained-ckpt '/home/pengguozhu/liuyidi/logdir/V4.1_4/V4.2_trans_256k_all_drop0.3_two_trans_64k/ckpt/checkpoint-iter-044000.pyth' \
--ckpt-log-dir "/home/pengguozhu/liuyidi/logdir/V4.1_4/V4.2_trans_256k_all_drop0.3_two_trans_64k_fix" \
--ckpt-save-interval 1000 \
--rand-corner 0 \
--fp16 0 \
--optimizer "adamw" \
--warmup_step 1000 \
--cos_step 16000 \
--only_trans 0 \
--fixres_2 1 

DIRNAME=/home/pengguozhu/liuyidi/logdir/V4.1_4/V4.2_trans_256k_all_drop0.3_two_trans_64k_fix
Nckpts=16000
Ninterval=1000
for ((i= 0; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    horovodrun -np 8 /data1/liuyidi/moco/four_pic/predict_hago_4_linear2.py \
    --list-file "/data1/liuyidi/scene_cls/V4.1/argu_list/train_list/val*4_list.txt" \
    --root-dir "/data1/liuyidi/scene_cls/6b_2/" \
    --load-npy 0 \
    --num-classes 43 \
    --net "resnest50" \
    --img-size 256 \
    --batch-size 64 \
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
