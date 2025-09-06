import torch
import torch.nn as nn
import torchvision.models as models
import os

from torch import device
from torch.utils.tensorboard import SummaryWriter


# import torch.nn.functional as F
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "./"))

class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.noise_reduction = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.global_avg_pool(x))
        max_out = self.fc(self.global_max_pool(x))
        combined_out = avg_out + max_out
        filtered_out = self.noise_reduction(combined_out)  # 引入语义降噪
        return x * self.sigmoid(filtered_out)


class EnhancedSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(EnhancedSpatialAttention, self).__init__()
        padding = kernel_size // 2
        #高斯滤波
        self.gaussian_blur = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.gaussian_blur.weight.data.fill_(1 / 25.0)  # 初始化为均值滤波

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 引入语义降噪模块
        self.noise_reduction = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim=1)
        spatial_attention_map = self.conv(pool_out)
        spatial_attention_map = self.gaussian_blur(spatial_attention_map)
        attention_weights = self.noise_reduction(spatial_attention_map)  # 应用语义降噪
        return x * self.sigmoid(attention_weights)


# 带通道和空间注意力的 ResNet50
class ResNet50WithCBAM(nn.Module):
    def __init__(self, num_classes, reduction_ratio=16):
        super(ResNet50WithCBAM, self).__init__()
        self.num_classes = num_classes

        self.resnet50 = models.resnet50(weights=None)
        load_path = os.path.join(root_dir, 'pretrained/resnet50.pth')
        self.resnet50.load_state_dict(torch.load(load_path, weights_only=True))

        feas = list(self.resnet50.children())[:-1]
        self.pre_layer = nn.Sequential(*feas[0:4])
        self.stage_1 = nn.Sequential(*feas[4])  # ResNet50 Stage1
        self.stage_2 = nn.Sequential(*feas[5])  # ResNet50 Stage2
        self.stage_3 = nn.Sequential(*feas[6])  # ResNet50 Stage3
        self.stage_4 = nn.Sequential(*feas[7])  # ResNet50 Stage4
        self.avg = feas[8]
        self.flatten = nn.Flatten()
        # 定义注意力模块
        self.channel_attention_1 = ChannelAttention(256, ratio=reduction_ratio)  # Stage 3
        self.spatial_attention_1 = EnhancedSpatialAttention(kernel_size=7)
        self.channel_attention_2 = ChannelAttention(512, ratio=reduction_ratio)  # Stage 3
        self.spatial_attention_2 = EnhancedSpatialAttention(kernel_size=7)

        self.channel_attention_3 = ChannelAttention(1024, ratio=reduction_ratio)  # Stage 3 (1024 channels)
        self.spatial_attention_3 = EnhancedSpatialAttention(kernel_size=7)
        self.channel_attention_4 = ChannelAttention(2048, ratio=reduction_ratio)  # Stage 4 (2048 channels)
        self.spatial_attention_4 = EnhancedSpatialAttention(kernel_size=7)

        # 最后的池化和全连接层
        self.avg_pool = self.resnet50.avgpool
        self.fc = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        x = self.pre_layer(x)  # 初始层（Stem + Stage1）[1,3,448,448]

        x = self.stage_1(x)  # Stage1[1,64,112,112]
        x = self.channel_attention_1(x)
        x = self.spatial_attention_1(x)

        x = self.stage_2(x)  # Stage2[1,256,112,112]
        x = self.channel_attention_2(x)
        x = self.spatial_attention_2(x)

        x = self.stage_3(x)  # Stage3[1,512,56,56]
        # 加入通道注意力和空间注意力
        x = self.channel_attention_3(x)#[1,1024,28,28]
        x = self.spatial_attention_3(x)#[1,1024,28,28]

        x = self.stage_4(x)  # Stage4#[1,1024,28,28]

        # 加入通道注意力和空间注意力
        x = self.channel_attention_4(x)#[1,2048,14,14]
        x = self.spatial_attention_4(x)#[1,2024,14,14]

        x = self.avg(x)  # 平均池化#[1,2024,14,14]
        x = self.flatten(x)  # 扁平化#[1,2024,1,1]
        x = self.fc(x)  # 全连接层
        return x


# 测试模型
if __name__ == "__main__":
    num_classes = 13

    model = ResNet50WithCBAM(num_classes=num_classes).cuda()
    inputs = torch.randn(1, 3, 448, 448).cuda()
    outputs = model(inputs)
    print(model)
    print("输出维度: ", outputs.size())
