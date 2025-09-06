import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.model_zoo as model_zoo

from torchvision.models import resnet18, resnet34, resnet101, resnet152

# 预训练模型链接
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############################################################################
# RegularLoss (只保留一个版本)
###############################################################################
class RegularLoss(nn.Module):
    def __init__(self, gamma=0, part_features=None, nparts=1):
        """
        Regularization loss for parts features.
        """
        super(RegularLoss, self).__init__()
        self.register_buffer('part_features', part_features)
        self.nparts = nparts
        self.gamma = gamma

    def forward(self, x):
        # x 应为 list，其中每个元素形状为 [N, D]（经过 squeeze）
        assert isinstance(x, list), "parts features should be presented in a list"
        corr_matrix = torch.zeros(self.nparts, self.nparts, device=x[0].device)
        for i in range(self.nparts):
            x[i] = x[i].squeeze()
            x[i] = torch.div(x[i], x[i].norm(dim=1, keepdim=True) + 1e-8)
        for i in range(self.nparts):
            for j in range(self.nparts):
                corr_matrix[i, j] = torch.mean(torch.mm(x[i], x[j].t()))
                if i == j:
                    corr_matrix[i, j] = 1.0 - corr_matrix[i, j]
        regloss = torch.mul(torch.sum(torch.triu(corr_matrix)), self.gamma)
        return regloss

###############################################################################
# SELayer
###############################################################################
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

###############################################################################
# MELayer (用于生成多个部分的ME输出)
###############################################################################
class MELayer(nn.Module):
    def __init__(self, channel, reduction=16, nparts=1):
        super(MELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.nparts = nparts
        parts = []
        for _ in range(self.nparts):
            parts.append(nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
            ))
        self.parts = nn.ModuleList(parts)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        meouts = []
        for i in range(self.nparts):
            meouts.append(x * self.parts[i](y).view(b, c, 1, 1))
        return meouts

###############################################################################
# Bottleneck 模块
###############################################################################
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, meflag=False, nparts=1, reduction=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)
        self.relu  = nn.ReLU(inplace=True)
        self.meflag = meflag
        if self.meflag:
            self.me = MELayer(planes * 4, nparts=nparts, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.meflag:
            outreach = out.clone()
            parts = self.me(outreach)
            out += residual
            out = self.relu(out)
            for i in range(len(parts)):
                parts[i] = self.relu(parts[i] + residual)
            # 返回 tuple (out, parts)
            return out, parts
        else:
            out += residual
            out = self.relu(out)
            return out

###############################################################################
# ResNet 模型定义
###############################################################################
class ResNet(nn.Module):
    def __init__(self, block, layers, nparts=1, meflag=False, num_classes=1000):
        self.nparts = nparts
        self.nclass = num_classes
        self.meflag = meflag
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # layer3 和 layer4 中传入 meflag 和 nparts 参数
        self.layer3 = self._make_layer(block, 256, layers[2], meflag=meflag, stride=2, nparts=nparts, reduction=256)
        self.layer4 = self._make_layer(block, 512, layers[3], meflag=meflag, stride=2, nparts=nparts, reduction=256)
        self.adpavgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_ulti = nn.Linear(512 * block.expansion * nparts, num_classes)

        if self.nparts > 1:
            self.adpmaxpool = nn.AdaptiveMaxPool2d(1)
            self.fc_plty = nn.Linear(256 * block.expansion * nparts, num_classes)
            self.fc_cmbn = nn.Linear(256 * block.expansion * nparts, num_classes)
            self.conv2_1 = nn.Conv2d(512 * block.expansion, 256 * block.expansion, kernel_size=1, bias=False)
            self.conv2_2 = nn.Conv2d(512 * block.expansion, 256 * block.expansion, kernel_size=1, bias=False)
            self.conv3_1 = nn.Conv2d(256 * block.expansion, 256 * block.expansion, kernel_size=3, padding=1, bias=False)
            self.conv3_2 = nn.Conv2d(256 * block.expansion, 256 * block.expansion, kernel_size=3, padding=1, bias=False)
            self.bn3_1 = nn.BatchNorm2d(256 * block.expansion)
            self.bn3_2 = nn.BatchNorm2d(256 * block.expansion)
            if nparts == 3:
                self.conv2_3 = nn.Conv2d(512 * block.expansion, 256 * block.expansion, kernel_size=1, bias=False)
                self.conv3_3 = nn.Conv2d(256 * block.expansion, 256 * block.expansion, kernel_size=3, padding=1, bias=False)
                self.bn3_3 = nn.BatchNorm2d(256 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, meflag=False, stride=1, nparts=1, reduction=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == blocks - 1:
                layers.append(block(self.inplanes, planes, meflag=meflag, nparts=nparts, reduction=reduction))
            else:
                layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        if self.meflag:
            # layer3 返回 tuple (x, plty_parts)
            x, plty_parts = self.layer3(x)
            # layer4 返回 tuple (_, ulti_parts)
            _, ulti_parts = self.layer4(x)
            cmbn_ftres = []
            for i in range(self.nparts):
                # 动态获取 plty_parts[i] 的空间尺寸作为目标尺寸
                target_size = plty_parts[i].shape[-2:]
                if i == 0:
                    ulti_parts_iplt = F.interpolate(self.conv2_1(ulti_parts[i]), size=target_size)
                    cmbn_ftres.append(self.adpavgpool(self.bn3_1(self.conv3_1(plty_parts[i] + ulti_parts_iplt))))
                elif i == 1:
                    ulti_parts_iplt = F.interpolate(self.conv2_2(ulti_parts[i]), size=target_size)
                    cmbn_ftres.append(self.adpavgpool(self.bn3_2(self.conv3_2(plty_parts[i] + ulti_parts_iplt))))
                elif i == 2:
                    ulti_parts_iplt = F.interpolate(self.conv2_3(ulti_parts[i]), size=target_size)
                    cmbn_ftres.append(self.adpavgpool(self.bn3_3(self.conv3_3(plty_parts[i] + ulti_parts_iplt))))
                plty_parts[i] = self.adpavgpool(plty_parts[i])
                ulti_parts[i] = self.adpavgpool(ulti_parts[i])
            # 将 ulti_parts 进行拼接，并经过全连接层
            xf = torch.cat(ulti_parts, 1)
            xf = xf.view(xf.size(0), -1)
            xf = self.fc_ulti(xf)
            # 修改：只返回 xf，确保输出形状为 [batch, num_classes]
            return xf
        else:
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.adpavgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc_ulti(x)
            return x

###############################################################################
# 提供接口函数构造模型（例如 ResNet-34 带 OSME）
###############################################################################
def resnet34_osme(num_classes, pretrained=False, modelpath=None, nparts=2, **kwargs):
    """
    构造一个基于 ResNet-34 并带有 OSME 机制的模型。
    :param num_classes: 分类数
    :param pretrained: 是否加载预训练权重
    :param modelpath: 预训练权重的路径（可选）
    :param nparts: 分支数（nparts>1 时自动打开 meflag）
    """
    kwargs.setdefault('nparts', nparts)
    if nparts > 1:
        kwargs.setdefault('meflag', True)
    else:
        kwargs.setdefault('meflag', False)
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    if pretrained:
        if modelpath is None:
            modelpath = model_urls['resnet34']
        state_dict = model_zoo.load_url(modelpath)
        model.load_state_dict(state_dict, strict=False)
    return model

###############################################################################
# 如果作为独立模块测试
###############################################################################
if __name__ == '__main__':
    # 测试 resnet34_osme，期望输出形状为 torch.Size([1, 13])
    model = resnet34_osme(num_classes=13, pretrained=False, nparts=2)
    model.eval()
    x = torch.randn(1, 3, 448, 448)
    output = model(x)
    print(output.shape)  # 应输出: torch.Size([1, 13])
