import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms

##############################################
# BasicConv 定义
##############################################
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

##############################################
# Features 模块，将 backbone 分为8个阶段
##############################################
class Features(nn.Module):
    def __init__(self, net_layers):
        super(Features, self).__init__()
        self.net_layer_0 = nn.Sequential(*net_layers[0])
        self.net_layer_1 = nn.Sequential(*net_layers[1])
        self.net_layer_2 = nn.Sequential(*net_layers[2])
        self.net_layer_3 = nn.Sequential(*net_layers[3])
        self.net_layer_4 = nn.Sequential(*net_layers[4])
        self.net_layer_5 = nn.Sequential(*net_layers[5])
        self.net_layer_6 = nn.Sequential(*net_layers[6])
        self.net_layer_7 = nn.Sequential(*net_layers[7])

    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)
        x = self.net_layer_3(x)
        x = self.net_layer_4(x)
        x1 = self.net_layer_5(x)
        x2 = self.net_layer_6(x1)
        x3 = self.net_layer_7(x2)
        return x1, x2, x3

##############################################
# Network_Wrapper 模块
##############################################
class Network_Wrapper(nn.Module):
    def __init__(self, net_layers, num_classes):  # Changed parameter name to num_classes
        super(Network_Wrapper, self).__init__()
        self.Features = Features(net_layers)
        # Three branches corresponding to different feature pooling sizes
        self.max_pool1 = nn.AdaptiveMaxPool2d((1, 1))
        self.max_pool2 = nn.AdaptiveMaxPool2d((1, 1))
        self.max_pool3 = nn.AdaptiveMaxPool2d((1, 1))

        # Adjusted conv_block1 based on VGG16_bn output channels
        self.conv_block1 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes)
        )

        # For conv_block2: x2 input channels corrected to 512
        self.conv_block2 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes)
        )

        # For conv_block3: assume x3 output channels as 512
        self.conv_block3 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes)
        )

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x1, x2, x3 = self.Features(x)

        x1_ = self.conv_block1(x1)
        map1 = x1_.detach()
        x1_ = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)
        x1_c = self.classifier1(x1_f)

        x2_ = self.conv_block2(x2)
        map2 = x2_.detach()
        x2_ = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)

        x3_ = self.conv_block3(x3)
        map3 = x3_.detach()
        x3_ = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier3(x3_f)

        x_c_all = torch.cat((x1_f, x2_f, x3_f), -1)
        x_c_all = self.classifier_concat(x_c_all)

        return x_c_all


##############################################
# 接口函数：构造 Network_Wrapper 模型（修正后的层分割）
##############################################
def build_network_wrapper(num_classes):
    backbone = models.vgg16_bn(pretrained=False).features
    indices = [
        0,  # conv1
        6,  # conv2
        13, # conv3
        23, # conv4
        33, # conv5
        43, # conv6 (assuming last conv layer index is 43)
        43, # No subsequent layers, use Identity
        43  # Same as above
    ]
    net_layers = []
    for i in range(len(indices) - 1):
        start, end = indices[i], indices[i+1]
        if end > start:
            net_layers.append([backbone[j] for j in range(start, end)])
        else:
            net_layers.append([nn.Identity()])
    while len(net_layers) < 8:
        net_layers.append([nn.Identity()])
    model = Network_Wrapper(net_layers, num_classes)
    return model


##############################################
# 测试代码
##############################################
if __name__ == '__main__':
    model = build_network_wrapper(num_classes=13)
    model.eval()
    x = torch.randn(1, 3, 448, 448)
    outputs = model(x)
    print("Output shapes:")
    for i, out in enumerate(outputs[:4]):  # 仅打印分类输出
        print(f"Output {i+1}: {out.shape}")