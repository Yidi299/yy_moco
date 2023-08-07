import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.init as init
import torch.optim as optim

class ImageTransformer2(nn.Module):
    def __init__(self, pretrain_model ,num_classes=43, d_model=2048, nhead=8, num_layers=2,dropout_rate = 0.1):
        super(ImageTransformer2, self).__init__()
        self.feature_extractor = pretrain_model
        # self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.dropout = nn.Dropout(dropout_rate)

        self.position_encoding = nn.Parameter(torch.randn(1, 4, d_model))
        
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)

        init.xavier_uniform_(self.fc.weight)
        init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        batch_size, timesteps, c, h, w = x.size()
        x = x.view(batch_size * timesteps, c, h, w)
        features = self.feature_extractor(x).squeeze()
        features = self.dropout(features)
        features = features.view(batch_size, timesteps, -1)
        
        features += self.position_encoding
        features = features.transpose(0, 1)
        features = self.transformer_encoder(features)

        
        features = features.transpose(0, 1)
        features = features[:, -1] ##提取最后一个时间步的输出
        
        out = self.fc(features)


        return out


class ImageTransformer(nn.Module):
    def __init__(self, pretrain_model ,num_classes=43, d_model=2048, nhead=8, num_layers=2):
        super(ImageTransformer, self).__init__()
        self.feature_extractor = pretrain_model
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        
        self.position_encoding = nn.Parameter(torch.randn(1, 4, d_model))
        
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)

        init.xavier_uniform_(self.fc.weight)
        init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        batch_size, timesteps, c, h, w = x.size()
        x = x.view(batch_size * timesteps, c, h, w)
        features = self.feature_extractor(x)

        features = features.view(batch_size, timesteps, -1)
        
        features += self.position_encoding
        features = features.transpose(0, 1)
        features = self.transformer_encoder(features)

        
        features = features.transpose(0, 1)
        features = features[:, -1] ##提取最后一个时间步的输出
        
        out = self.fc(features)


        return out
    
if __name__ == '__main__':
    input = torch.randn(16, 4, 3, 224, 224)
    model = ImageTransformer2(resnet50(pretrained=False))
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    output = model(input)



    


    parameters_trans = []
    for name, param in model.named_parameters():
            if not name.startswith('feature_extractor'):
                
                parameters_trans.append(param)
    
    
    bn_params = [v for n, v in model.feature_extractor.named_parameters() if ('bn' in n or 'bias' in n)]
    rest_params = [v for n, v in model.feature_extractor.named_parameters() if not ('bn' in n or 'bias' in n)]

    trans_params = [v for n, v in model.named_parameters() if not ('feature_extractor' in n )]



    optimizer = optim.SGD(model.parameters(), lr=0.01)


s = 0