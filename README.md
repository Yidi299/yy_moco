#swin_base 训练
* sdad
```
horovodrun -np 8 python hago/train_hago_timm.py \
--list-file "/data1/liuyidi/scene_cls/V4.1.2/list/train_list/train_list_remove4_new_resample.txt" \
--root-dir "/data1/liuyidi/scene_cls/6b_2/" \
--batch-size 8 \
--img-size 256 \
--base-lr 2e-5 \
--net "timm.swinv2_base_window12to16_192to256_22kft1k" \
--sam 0 \
--mixup 0.8 \
--ckpt-log-dir '/data1/liuyidi/scene_cls/V4.1.2/log_dir/V4.1.2_test37_38_remove_act=8' \
--pretrained-ckpt '/data1/liuyidi/model/swin_V2/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth' \  
--final-drop 0.2 \
--rand-corner 1 \
--label_smoothing 0.1 \
--weight-decay 1e-8 \
--ckpt-save-interval 4000 \
--optimizer 'adamw' \
--accumulation_steps 16 \
--cos_step 1024000 \
--base_batchsize 1024 \
--warmup_step 64000 
```

#swin_base 验证
```
DIRNAME=/data1/liuyidi/scene_cls/V4.1.2/log_dir/V4.1.2_test37_38_remove_swin_base_22kft1k
Nckpts=512000
Ninterval=8000
for ((i=512000; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    horovodrun -np 7 python hago/predict_hago_timm.py \
    --list-file "/data1/liuyidi/scene_cls/V4.1.2/list/train_list/val_merge.txt" \
    --root-dir "/data1/liuyidi/scene_cls/6b_2/" \
    --load-npy 0 \
    --num-classes 43 \
    --net "timm.swinv2_base_window12to16_192to256_22kft1k" \
    --img-size 256 \
    --batch-size 64 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    --out-per-n 10000 \
    --out "$DIRNAME/val/val-$part_n"
    python hago/val_hago.py \
    "/data1/liuyidi/scene_cls/V4.1.2/list/train_list/val_merge.txt" \
    "$DIRNAME/val/val-$part_n" \
    "${DIRNAME}/log/val/" \
    ${i} &
```

#swin_base fixres训练
```
horovodrun -np 8 python hago/train_hago_timm.py \
--list-file "/data1/liuyidi/scene_cls/V4.1.2/list/train_list/train_list_remove4_new_resample.txt" \
--root-dir "/data1/liuyidi/scene_cls/6b_2/" \
--batch-size 4 \
--num-classes 43 \
--img-size 384 \
--fixres 1 \
--base-lr 1e-4 \
--sam 1 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 0.05 \
--net "setting" \
--pretrained-ckpt "/data1/liuyidi/scene_cls/V4.1.2/log_dir/swin_base_22kft1k_1024k/ckpt/checkpoint-iter-1024000.pyth" \
--ckpt-log-dir "/data1/liuyidi/scene_cls/V4.1.2/log_dir/swin_base_22kft1k_1024k_fix_128k_cpb_2" \
--ckpt-save-interval 4000 \
--label_smoothing 0.1 \
--optimizer 'adamw' \
--cos_step 128000 \
--warmup_step 8000 \
--fp16 0 \
--accumulation_steps 32
```
#swin_base fixres 验证
```
DIRNAME=/data1/liuyidi/scene_cls/V4.1.2/log_dir/swin_base_22kft1k_1024k_fix_128k_cpb
Nckpts=128000
Ninterval=4000
for ((i= 128000 ; i<=$Nckpts; i+=$Ninterval)); do
    part_n=`printf "%06d" ${i}`
    horovodrun -np 6 python hago/predict_hago_timm.py \
    --list-file "/data1/liuyidi/scene_cls/V4.1.2/list/train_list/val_1682.txt" \
    --root-dir "/data1/liuyidi/scene_cls/6b_2/" \
    --load-npy 0 \
    --num-classes 43 \
    --net "setting" \
    --img-size 384 \
    --batch-size 32 \
    --ckpt "$DIRNAME/ckpt/checkpoint-iter-${part_n}.pyth" \
    --out-per-n 10000 \
    --out "$DIRNAME/val/val-$part_n"
    python hago/val_hago.py \
    "/data1/liuyidi/scene_cls/V4.1.2/list/train_list/val_1682.txt" \
    "$DIRNAME/val/val-$part_n" \
    "${DIRNAME}/log/val/" \
    ${i} &
done
```

四图特征融合训练
```
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
```

四图特征融合验证

```
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
```





