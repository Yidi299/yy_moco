import torch 


ck  = torch.load('/data1/liuyidi/scene_cls/V4.1/log_dir/V4.2_duibi3/ckpt/checkpoint-iter-002000.pyth')['model_state']
ck2 = torch.load('/data1/liuyidi/scene_cls/V4.1/log_dir/V4.1_test27_1_fix/ckpt/checkpoint-iter-008000.pyth')['model_state']

for i,j in zip(ck.keys(),ck2.keys()):
        if torch.equal(ck[i].data,ck2[j].data): ##判断两个tensor是否相等使用torch.equal
            print(i,'same')
        else:
            print(i,'not same')
            print(i,ck[i].data)

s = 0