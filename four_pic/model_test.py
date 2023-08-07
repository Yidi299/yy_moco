import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import resnet50

class CustomModel3(nn.Module):
    def __init__(self, feature_extractor, num_classes = 43 ,input_features =2048, dropout_rate=0.5):
        super(CustomModel3, self).__init__()
        
        ## 用于提取特征的预训练模型,去掉最后一层

        self.feature_extractor =  nn.Sequential(*list(feature_extractor.children())[:-1]) 
        self.feature_extractor.eval()

        
        self.con_1d_layer = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(input_features , num_classes)

        init.xavier_uniform_(self.fc.weight)
        init.constant_(self.fc.bias, 0.0)
        
    def forward(self, imgs):
        print(imgs.shape)
        # 假设imgs的形状为 (batch_size, 4, C, H, W)
        batch_size, _, C, H, W = imgs.shape
        imgs0 = imgs.view(-1, C, H, W)  # 将图片展开为 (batch_size * 4, C, H, W)
        
        with torch.no_grad():
            features = self.feature_extractor(imgs0)  # 提取特征 (batch_size * 4, output_dim)
        
        features_grouped = features.view(batch_size,4, -1)  # 将特征按组分组 (batch_size, output_dim * 4)
        
        features_grouped = self.con_1d_layer(features_grouped)  # 输出 (batch_size, output_dim)

        features_grouped = features_grouped.view(batch_size, -1)  

        features_grouped = self.dropout(features_grouped)  # 添加 dropout 

        out = self.fc(features_grouped)  # 输出 (batch_size, num_classes)


        # images_ls = torch.unbind(imgs, dim=1)
        # imgs_2 = images_ls[-1]

        # imgs_2 = imgs[:,3,:,:,:]
        # with torch.no_grad():
        #     out = self.feature_extractor(imgs_2) 

        # fea0 = features_grouped[:,0,:]
        # out =  self.fc(fea0)
        
        return out


class CustomModel4(nn.Module):
    def __init__(self, feature_extractor, num_classes = 43 ,input_features =2048, dropout_rate=0.5):
        super(CustomModel4, self).__init__()
        
        ## 用于提取特征的预训练模型,去掉最后一层

        self.feature_extractor =  nn.Sequential(*list(feature_extractor.children())[:-1]) 
        self.con_1d_layer = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(input_features , num_classes)

        init.xavier_uniform_(self.fc.weight)
        init.constant_(self.fc.bias, 0.0)
        
    def forward(self, imgs):
        print(imgs.shape)
        # 假设imgs的形状为 (batch_size, 4, C, H, W)
        batch_size, _, C, H, W = imgs.shape
        imgs0 = imgs.view(-1, C, H, W)  # 将图片展开为 (batch_size * 4, C, H, W)
        
        
        features = self.feature_extractor(imgs0)  # 提取特征 (batch_size * 4, output_dim)
        
        features_grouped = features.view(batch_size,4, -1)  # 将特征按组分组 (batch_size, output_dim * 4)
        
        features_grouped = self.con_1d_layer(features_grouped)  # 输出 (batch_size, output_dim)

        features_grouped = features_grouped.view(batch_size, -1)  

        features_grouped = self.dropout(features_grouped)  # 添加 dropout 

        out = self.fc(features_grouped)  # 输出 (batch_size, num_classes)
        
        return out
    

if __name__ == '__main__':
    input = torch.randn(16, 4, 3, 224, 224)
    model = CustomModel4(resnet50(pretrained=False))


    

    output = model(input)

    parameters_name =  []
    for name, param in model.named_parameters():
            # if name.startswith('feature_extractor'):
            #     param.requires_grad = False
            # else:
            #     parameters_name.append(name)
        parameters_name.append(name)
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

s = 0