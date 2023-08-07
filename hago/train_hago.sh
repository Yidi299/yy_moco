horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/diaoxing-img-train.txt.1" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 3 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 8000,14000,20000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/diaoxing-0705-moco1-init" \
--ckpt-save-interval 100 \
--fp16 1



horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/testset/train_0617_all.txt.sex01" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 32 \
--num-classes 5 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 8000,14000,20000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 320 \
--ckpt-log-dir "ckpt-hago/sex-0617-moco1-init-tmp" \
--ckpt-save-interval 100000 \
--fp16 1




for d in sex-0617-* ; do
    a=`ll $d/ckpt/checkpoint-iter-*.pyth | tail -1 | awk '{print $9}'` 
    mv $a .
    rm -rf $d/ckpt/checkpoint-iter-*.pyth
    mv checkpoint-iter-*.pyth $a
done


cat ../hago/testset/train_0617_all.txt.1 | \
awk '{if($2==0 || $2==1 || $2==4) {for (i=0; i<5; ++i) print $0} else print $0}' \
> ../hago/testset/train_0617_all.txt.mining26w
for ((i=0; i<5; ++i)); do
    cat ../hago/tag-by-model/mining-tagged-26w.txt | awk '{print substr($0, 37)}' \
    >> ../hago/testset/train_0617_all.txt.mining26w
done


cat ../testset/train_0617_all.txt.1 | \
awk '{if($2==0) {for (i=0; i<5; ++i) print $0} else if($2==1) print $0}' \
> ../testset/train_0617_all.txt.sex01
cat mining-tagged-60w.txt | \
awk '{if($2==0) {for (i=0; i<5; ++i) print $0} else if($2==1) print $0}' | \
awk '{print substr($0, 37)}' \
>> ../testset/train_0617_all.txt.sex01


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/testset/train_0617_all.txt.sex01" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 5 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 8000,14000,20000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/sex-0617-moco1-init-sex01" \
--ckpt-save-interval 100 \
--fp16 1




horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/testset/train_0617_all.txt.mining26w" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 5 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 8000,14000,20000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/sex-0617-moco1-init-mining-26w" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/testset/train_0617_all.txt.mining26w" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 5 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 16000,28000,40000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/sex-0617-moco1-init-mining-26w-2" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/testset/train_0617_all.txt.mining26w" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 5 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 40000,70000,100000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/sex-0617-moco1-init-mining-26w-3" \
--ckpt-save-interval 500 \
--fp16 1






# 原始分布
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/testset/train_0617_all.txt.1" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 5 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 8000,14000,20000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 5e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/sex-0617-moco1-init" \
--ckpt-save-interval 100 \
--fp16 1


# 性感一般过采样10倍
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/testset/train_0617_all.txt.2" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 5 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 8000,14000,20000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 5e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/sex-0617-moco1-init-sex1-x10" \
--ckpt-save-interval 100 \
--fp16 1

# python hago/val_hago.py ckpt-hago/sex-0617-moco1-init-sex1-x10/ckpt/checkpoint-iter-011400.pyth


    # 正则化 1
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/testset/train_0617_all.txt.2" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 5 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 8000,14000,20000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/sex-0617-moco1-init-sex1-x10-1" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/testset/train_0617_all.txt.2" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 5 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 15000,20000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/sex-0617-moco1-init-sex1-x10-1b" \
--ckpt-save-interval 100 \
--fp16 1

    # 正则化 2
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/testset/train_0617_all.txt.2" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 5 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 8000,14000,20000 \
--mixup 0.0 \
--final-drop 0.0 \
--no-bias-bn-wd 0 \
--weight-decay 5e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/sex-0617-moco1-init-sex1-x10-2" \
--ckpt-save-interval 100 \
--fp16 1


# 加入性感原因细分类数据
horovodrun -np 8 python hago/train_hago.py \
--list-file "data/train_0617_all.fine.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 5 \
--fine-grained 1 \
--fine-grained-w 1.0 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 8000,14000,20000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 5e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/sex-0617-moco1-init-sex1-x10-grained" \
--ckpt-save-interval 100 \
--fp16 1



# cat ../testset/train_0617_all.txt.2 > ../testset/train_0617_all.txt.3
# for ((i=0; i<10; ++i)); do
#     cat mining-tagged-10w.txt | awk '{print substr($0, 37)}' >> ../testset/train_0617_all.txt.3
# done
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/testset/train_0617_all.txt.3" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 5 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 8000,14000,20000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/sex-0617-moco1-init-add-mining-10w" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/testset/train_0617_all.txt.3" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 5 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 14000,20000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/sex-0617-moco1-init-add-mining-10w-b" \
--ckpt-save-interval 100 \
--fp16 1


# horovodrun -np 8 python hago/predict_hago.py ckpt-hago/sex-0617-moco1-init-add-mining-10w/ckpt/checkpoint-iter-012900.pyth 



# @葆明数据
horovodrun -np 8 python hago/train_hago.py \
--list-file "train_second_20200618_data.txt" \
--root-dir  " " \
--batch-size 64 \
--num-classes 5 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.003 \
--lr-stages-step 8000,14000,20000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/sex-0617-moco1-init-for-baoming" \
--ckpt-save-interval 100 \
--fp16 1






# 0810 调性
horovodrun -np 4 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/diaoxing-tra-08010.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 128 \
--num-classes 3 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 3000,5000,7000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/diaoxing-0810-moco1-init" \
--ckpt-save-interval 50 \
--fp16 1


horovodrun -np 4 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/diaoxing-tra-08010-add-img2.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 128 \
--num-classes 3 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 3000,5000,7000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/diaoxing-0810-add-img2-moco1-init" \
--ckpt-save-interval 50 \
--fp16 1




horovodrun -np 4 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-08010.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 128 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 6000,10000,13000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-0810-moco1-init" \
--ckpt-save-interval 50 \
--fp16 1

horovodrun -np 4 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-08010-add-img2.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 128 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 6000,10000,13000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-0810-add-img2-moco1-init" \
--ckpt-save-interval 50 \
--fp16 1



cd /data/remote/hago/daily-tag
python mk_data.py && python mk_data_balance.py dddd.txt data-train/pinlei-tra-08020.txt && rm -rf dddd.txt

# 0820
horovodrun -np 4 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-08020.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 12000,20000,26000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-0820-moco1-init" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/diaoxing-tra-08020.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 3 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 3000,5000,7000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/diaoxing-0820-moco1-init" \
--ckpt-save-interval 50 \
--fp16 1




horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/nianling-tra-0820.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 7 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 6000,10000,13000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/nianling-0820-moco1-init" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/nianling-tra-0820-img2.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 7 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 6000,10000,13500 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/nianling-0820-img2-moco1-init" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/yanzhi-tra-0820.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 3 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 6000,10000,13000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/yanzhi-0820-moco1-init" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/renshu-tra-0820.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 2 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 6000,10000,13000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/renshu-0820-moco1-init" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/renshu-tra-0820-c4.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 4 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 6000,10000,13100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/renshu-0820-c4-moco1-init" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/renshu-tra-0820-c3.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 3 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 6000,10000,13100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/renshu-0820-c3-moco1-init" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/renshu-tra-0820-c4-img2.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 4 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 6000,10000,13100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/renshu-0820-c4-img2-moco1-init" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/renshu-tra-0820-c3-img2.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 3 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 6000,10000,13100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/renshu-0820-c3-img2-moco1-init" \
--ckpt-save-interval 100 \
--fp16 1



mkdir ckpt-hago/pinlei-0810-moco1-init/ret
for ((i=0; i<=125; i+=5)); do
n=`printf "%04d00" $i`
CUDA_VISIBLE_DEVICES=1 python hago/predict_hago.py \
ckpt-hago/pinlei-0810-moco1-init/ckpt/checkpoint-iter-$n.pyth 90 ckpt-hago/pinlei-0810-moco1-init/ret/ret-$n
done

mkdir ckpt-hago/pinlei-0810-add-img2-moco1-init/ret
for ((i=0; i<=125; i+=5)); do
n=`printf "%04d00" $i`
CUDA_VISIBLE_DEVICES=3 python hago/predict_hago.py \
ckpt-hago/pinlei-0810-add-img2-moco1-init/ckpt/checkpoint-iter-$n.pyth 90 \
ckpt-hago/pinlei-0810-add-img2-moco1-init/ret/ret-$n
done

DIR=ckpt-hago/nianling-0820-moco1-init
CLSN=7
mkdir $DIR/ret
for ((i=0; i<=15; i+=1)); do
n=`printf "%03d000" $i`
CUDA_VISIBLE_DEVICES=4 python hago/predict_hago.py $DIR/ckpt/checkpoint-iter-$n.pyth $CLSN $DIR/ret/ret-$n
done


######
# 0820
######
DIR=ckpt-hago/nianling-0820-img2-moco1-init
CLSN=7
mkdir $DIR/ret
for ((i=0; i<=15; i+=1)); do
n=`printf "%03d000" $i`
CUDA_VISIBLE_DEVICES=5 python hago/predict_hago.py $DIR/ckpt/checkpoint-iter-$n.pyth $CLSN $DIR/ret/ret-$n
done

DIR=ckpt-hago/nianling-0820-img2-moco1-init
CLSN=7
mkdir $DIR/ret
for ((i=0; i<=15; i+=1)); do
n=`printf "%03d000" $i`
CUDA_VISIBLE_DEVICES=5 python hago/predict_hago.py $DIR/ckpt/checkpoint-iter-$n.pyth $CLSN $DIR/ret/ret-$n
done

DIR=ckpt-hago/renshu-0820-moco1-init
CLSN=2
mkdir $DIR/ret
for ((i=0; i<=15; i+=1)); do
n=`printf "%03d000" $i`
CUDA_VISIBLE_DEVICES=6 python hago/predict_hago.py $DIR/ckpt/checkpoint-iter-$n.pyth $CLSN $DIR/ret/ret-$n
done

DIR=ckpt-hago/yanzhi-0820-moco1-init
CLSN=3
mkdir $DIR/ret
for ((i=0; i<=15; i+=1)); do
n=`printf "%03d000" $i`
CUDA_VISIBLE_DEVICES=7 python hago/predict_hago.py $DIR/ckpt/checkpoint-iter-$n.pyth $CLSN $DIR/ret/ret-$n
done

DIR=ckpt-hago/pinlei-0820-moco1-init
CLSN=90
mkdir $DIR/ret
for ((i=0; i<=25; i+=1)); do
n=`printf "%03d000" $i`
CUDA_VISIBLE_DEVICES=3 python hago/predict_hago.py $DIR/ckpt/checkpoint-iter-$n.pyth $CLSN $DIR/ret/ret-$n
done

DIR=ckpt-hago/diaoxing-0820-moco1-init
CLSN=3
mkdir $DIR/ret
for ((i=0; i<=65; i+=5)); do
n=`printf "%03d000" $i`
CUDA_VISIBLE_DEVICES=2 python hago/predict_hago.py $DIR/ckpt/checkpoint-iter-$n.pyth $CLSN $DIR/ret/ret-$n
done



DIR=ckpt-hago/renshu-0820-c4-moco1-init
CLSN=4
mkdir $DIR/ret
for ((i=0; i<=15; i+=1)); do
n=`printf "%03d000" $i`
CUDA_VISIBLE_DEVICES=0 python hago/predict_hago.py $DIR/ckpt/checkpoint-iter-$n.pyth $CLSN $DIR/ret/ret-$n
done

DIR=ckpt-hago/renshu-0820-c3-moco1-init
CLSN=3
mkdir $DIR/ret
for ((i=0; i<=15; i+=1)); do
n=`printf "%03d000" $i`
CUDA_VISIBLE_DEVICES=1 python hago/predict_hago.py $DIR/ckpt/checkpoint-iter-$n.pyth $CLSN $DIR/ret/ret-$n
done

DIR=ckpt-hago/renshu-0820-c4-img2-moco1-init
CLSN=4
mkdir $DIR/ret
for ((i=0; i<=15; i+=1)); do
n=`printf "%03d000" $i`
CUDA_VISIBLE_DEVICES=2 python hago/predict_hago.py $DIR/ckpt/checkpoint-iter-$n.pyth $CLSN $DIR/ret/ret-$n
done

DIR=ckpt-hago/renshu-0820-c3-img2-moco1-init
CLSN=3
mkdir $DIR/ret
for ((i=0; i<=15; i+=1)); do
n=`printf "%03d000" $i`
CUDA_VISIBLE_DEVICES=3 python hago/predict_hago.py $DIR/ckpt/checkpoint-iter-$n.pyth $CLSN $DIR/ret/ret-$n
done

DIR=ckpt-hago/renshu-0820-c4-moco1-init-2
CLSN=4
mkdir $DIR/ret
for ((i=0; i<=35; i+=1)); do
n=`printf "%03d000" $i`
CUDA_VISIBLE_DEVICES=7 python hago/predict_hago.py $DIR/ckpt/checkpoint-iter-$n.pyth $CLSN $DIR/ret/ret-$n
done


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/pinlei-tra-08010.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 8000,12000,160000 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-0810-moco1-init-22222" \
--ckpt-save-interval 10000 \
--fp16 1












# 1010-1018
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/data-zhiliang-tra-1018.b.txt" \
--root-dir  "http://filer.ai.yy.com:9889/dataset/heliangliang/hago/" \
--batch-size 64 \
--num-classes 3 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 6000,10000,13100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/zhiliang-1020-c3-img1-moco1-init" \
--ckpt-save-interval 100 \
--fp16 1

# 1001-1018
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/data-zhiliang-tra-1018-2.b.txt" \
--root-dir  "http://filer.ai.yy.com:9889/dataset/heliangliang/hago/" \
--batch-size 64 \
--num-classes 3 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 6000,10000,13100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/zhiliang-1020-c3-img1-moco1-init-2" \
--ckpt-save-interval 100 \
--fp16 1

# 1001-1018, 多图贴
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/data-zhiliang-tra-1018-4.b.txt" \
--root-dir  "http://filer.ai.yy.com:9889/dataset/heliangliang/hago/" \
--batch-size 64 \
--num-classes 3 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 6000,10000,13100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/zhiliang-1020-c3-img1-moco1-init-2-img2" \
--ckpt-save-interval 100 \
--fp16 1

# 1010-1018, 多图贴
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/data-zhiliang-tra-1018-3.b.txt" \
--root-dir  "http://filer.ai.yy.com:9889/dataset/heliangliang/hago/" \
--batch-size 64 \
--num-classes 3 \
--pretrained-ckpt "ckpt-moco/ckpt-hago-1-rmdp-resnest-imgnetpre/checkpoint-iter-087499.pyth.bn" \
--base-lr 0.01 \
--lr-stages-step 6000,10000,13100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/zhiliang-1020-c3-img1-moco1-init-img2" \
--ckpt-save-interval 100 \
--fp16 1






horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/data-pinlei-tra-0801-1018.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
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
--ckpt-log-dir "ckpt-hago/pinlei-1020-moco1-init" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/data-pinlei-tra-0801-1008.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
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
--ckpt-log-dir "ckpt-hago/pinlei-1020-moco1-init-1008" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/data-pinlei-tra-0801-1008.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
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
--ckpt-log-dir "ckpt-hago/pinlei-1020-moco1-init-1008-2" \
--ckpt-save-interval 200 \
--fp16 1


horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/data-pinlei-tra-0801-1008.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
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
--ckpt-log-dir "ckpt-hago/pinlei-1020-imgnet-init-1008" \
--ckpt-save-interval 100 \
--fp16 1

horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/data-pinlei-tra-0801-1008.txt" \
--root-dir  "http://filer.ai.yy.com:9899/dataset/" \
--batch-size 64 \
--num-classes 90 \
--pretrained-ckpt "ckpt-moco/resnest101-22405ba7.pth" \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32100 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size 224 \
--ckpt-log-dir "ckpt-hago/pinlei-1020-imgnet-init-1008-2" \
--ckpt-save-interval 200 \
--fp16 1




CKPTDIR=ckpt-hago/zhiliang-1120-c3-img2-self-init
RES=224
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/data-zhiliang-tra-1120.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/heliangliang/hago/" \
--batch-size 64 \
--num-classes 3 \
--pretrained-ckpt "ckpt-hago/pinlei-1110-self-init-128k-0.03/ckpt/checkpoint-iter-128000.pyth" \
--base-lr 0.01 \
--lr-stages-step 16000,24000,32001 \
--mixup 0.0 \
--final-drop 0.2 \
--no-bias-bn-wd 1 \
--weight-decay 1e-4 \
--img-size $RES \
--ckpt-log-dir "${CKPTDIR}" \
--ckpt-save-interval 100 \
--fp16 1

CKPTDIR=ckpt-hago/zhiliang-1120-c3-img2-self-init
RES=320
horovodrun -np 8 python hago/train_hago.py \
--list-file "../hago/daily-tag/data-train/data-zhiliang-tra-1120.txt" \
--root-dir  "http://10.28.32.58:9200/dataset/heliangliang/hago/" \
--batch-size 64 \
--num-classes 3 \
--pretrained-ckpt "${CKPTDIR}/ckpt/checkpoint-iter-032000.pyth" \
--fixres 1 \
--final-drop 0.2 \
--base-lr 0.01 \
--lr-stages-step 8000,12000,16100 \
--img-size $RES \
--ckpt-log-dir "${CKPTDIR}-fixres-finetune-$RES-layer4" \
--ckpt-save-interval 100 \
--fp16 1


horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/data-zhiliang-val-1123.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/heliangliang/hago/" \
    --num-classes 3 \
    --net "resnest101" \
    --img-size 256 \
    --batch-size 32 \
    --ckpt "ckpt-hago/zhiliang-1020-c3-img1-moco1-init-2-img2/ckpt/checkpoint-iter-010000.pyth" \
    --out  "ckpt-hago/zhiliang-1020-c3-img1-moco1-init-2-img2/val/zhiliang-val-1123.txt-10"
python hago/val_hago.py \
    ../hago/daily-tag/data-val/data-zhiliang-val-1123.txt \
    ckpt-hago/zhiliang-1020-c3-img1-moco1-init-2-img2/val/zhiliang-val-1123.txt-10 \
    null 0

horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/data-zhiliang-val-1123.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/heliangliang/hago/" \
    --num-classes 3 \
    --net "resnest101" \
    --img-size 320 \
    --batch-size 32 \
    --ckpt "ckpt-hago/zhiliang-1120-c3-img2-moco1-init-fixres-finetune-320-layer4/ckpt/checkpoint-iter-016000.pyth" \
    --out  "ckpt-hago/zhiliang-1120-c3-img2-moco1-init-fixres-finetune-320-layer4/val/zhiliang-val-1123.txt-16"
python hago/val_hago.py \
    ../hago/daily-tag/data-val/data-zhiliang-val-1123.txt \
    ckpt-hago/zhiliang-1120-c3-img2-moco1-init-fixres-finetune-320-layer4/val/zhiliang-val-1123.txt-16 \
    null 0

horovodrun -np 8 python hago/predict_hago.py \
    --list-file "../hago/daily-tag/data-val/data-zhiliang-val-1123.txt" \
    --root-dir  "http://10.28.32.58:9200/dataset/heliangliang/hago/" \
    --num-classes 3 \
    --net "resnest101" \
    --img-size 320 \
    --batch-size 32 \
    --ckpt "ckpt-hago/zhiliang-1120-c3-img2-self-init-fixres-finetune-320-layer4/ckpt/checkpoint-iter-016000.pyth" \
    --out  "ckpt-hago/zhiliang-1120-c3-img2-self-init-fixres-finetune-320-layer4/val/zhiliang-val-1123.txt-16"
python hago/val_hago.py \
    ../hago/daily-tag/data-val/data-zhiliang-val-1123.txt \
    ckpt-hago/zhiliang-1120-c3-img2-self-init-fixres-finetune-320-layer4/val/zhiliang-val-1123.txt-16 \
    null 0

