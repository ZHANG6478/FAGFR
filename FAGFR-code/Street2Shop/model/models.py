
import torch
import torch.nn as nn
import torchvision.models as models
import os
# import torch.nn.functional as F
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "./"))

class Vgg(nn.Module):
    def __init__(self, num_classes):
        super(Vgg, self).__init__()
        self.Vgg16_bn = models.vgg16_bn(weights=None)
        load_path = os.path.join(root_dir, 'pretrained/vgg16_bn.pth')
        self.Vgg16_bn.load_state_dict(torch.load(load_path, weights_only=True))
        feas = list(self.Vgg16_bn.children())[:-1]
        self.conv_layers = nn.Sequential(*feas)
        self.flatten = nn.Flatten()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avg(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(weights=None)
        load_path = os.path.join(root_dir, 'pretrained/resnet50.pth')
        self.resnet50.load_state_dict(torch.load(load_path, weights_only=True))
        feas = list(self.resnet50.children())[:-1]
        self.conv_layers = nn.Sequential(*feas)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class Alex_Net(nn.Module):
    def __init__(self, num_classes):
        super(Alex_Net, self).__init__()
        self.num_classes = num_classes
        self.alex_net = models.AlexNet()
        load_path = os.path.join(root_dir, 'pretrained/Alexnet.pth')
        self.alex_net.load_state_dict(torch.load(load_path, weights_only=True))
        self.alex_net.classifier[-1] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.alex_net(x)
        return x


if __name__ == '__main__':
    img = torch.randn(1, 3, 448, 448)
    net = ResNet50(num_classes=13)
    # print(net(img).shape)
    print(net)
    print(net(img).shape)
