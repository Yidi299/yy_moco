import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.init as init



class TimeSeriesClassifier(nn.Module):
    def __init__(self, feature_extractor, num_classes =43, hidden_size = 2048, num_layers = 2):
        super(TimeSeriesClassifier, self).__init__()
        
        # 使用预训练的模型进行特征提取
        self.feature_extractor = feature_extractor
        # 移除预训练模型的分类层
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1]) 

        # LSTM用于处理时序数据
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # 分类层
        self.fc = nn.Linear(hidden_size, num_classes)

        init.xavier_uniform_(self.fc.weight)
        init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        # x.shape: (batch_size, time_steps, channels, height, width)
        batch_size, time_steps, channels, height, width = x.shape

        # 将输入数据重新整形为: (batch_size * time_steps, channels, height, width)
        x = x.view(batch_size * time_steps, channels, height, width)

        # 通过预训练的模型提取特征
        features = self.feature_extractor(x)
        # features.shape: (batch_size * time_steps, hidden_size)

        # 将特征重新整形为: (batch_size, time_steps, hidden_size)
        features = features.view(batch_size, time_steps, -1)

        # 使用LSTM处理时序特征
        output, _ = self.lstm(features)
        # output.shape: (batch_size, time_steps, hidden_size)

        # 取最后一个时间步的输出
        last_output = output[:, -1, :]
        # last_output.shape: (batch_size, hidden_size)

        # 通过分类层进行43分类
        predictions = self.fc(last_output)
        # predictions.shape: (batch_size, num_classes)

        return predictions

if __name__ == '__main__':
    feature_extractor = resnet50(pretrained=False)
    model = TimeSeriesClassifier(feature_extractor, num_classes=43, hidden_size=2048, num_layers=2)
    input = torch.randn(16, 4, 3, 224, 224)
    # output = model(input)
    
    parameters_name =  []
    for name, param in model.named_parameters():
            if name.startswith('feature_extractor'):
                param.requires_grad = False
            else:
                parameters_name.append(name)
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    optim_f = torch.optim.SGD
    optimizer = optim_f(parameters, lr=base_lr, momentum=0.9, weight_decay=args.weight_decay)
    s = 0