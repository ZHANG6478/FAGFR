import torch
import torch.nn as nn
import torchvision.models as models
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "./"))
# print(root_dir)


class BCNN(torch.nn.Module):
    def __init__(self, num_classes):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.

        load_path = os.path.join(root_dir, 'pretrained','vgg16_bn.pth')
        self.vgg_16 = models.vgg16_bn(weights=None)
        self.vgg_16.load_state_dict(torch.load(load_path, weights_only=True))
        self.features = self.vgg_16.features
        self.features = torch.nn.Sequential(*list(self.features.children())[:-1])  # Remove pool5.
        # Linear classifier.
        self.fc = torch.nn.Linear(512 ** 2, num_classes)

    def forward(self, x):
        bs = x.size(0)
        x = self.features(x)
        x = x.view(bs, 512, -1)

        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (28 ** 2)  # Bilinear
        x = x.view(bs, 512 ** 2)
        x = torch.sqrt(x + 1e-5)
        x = torch.nn.functional.normalize(x)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    img = torch.randn(1, 3, 448, 448)
    bcnn = BCNN(num_classes=13)
    fea = bcnn(img)
    print(fea.shape)
