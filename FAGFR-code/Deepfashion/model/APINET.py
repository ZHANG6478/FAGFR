import os
import torch
from torch import nn
import torchvision
from torchvision import models
import numpy as np
import torch.nn.functional as F

# 使用当前目录作为 root_dir
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "./"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)
    return distance_matrix

class API_Net(nn.Module):
    def __init__(self, num_classes):
        super(API_Net, self).__init__()
        model_path = os.path.join(root_dir, 'pretrained', 'resnet50.pth')
        resnet50 = models.resnet50(weights=None)

        # 加载预训练权重，并删除 fc 层参数，避免尺寸不匹配
        state_dict = torch.load(model_path, weights_only=True)
        if 'fc.weight' in state_dict:
            del state_dict['fc.weight']
        if 'fc.bias' in state_dict:
            del state_dict['fc.bias']
        resnet50.load_state_dict(state_dict, strict=False)
        # 只使用除最后两层之外的部分（不包含 fc 层）
        layers = list(resnet50.children())[:-2]
        self.conv = nn.Sequential(*layers)
        # 修改为 kernel_size=7 以适应 7x7 的特征图
        self.avg = nn.AvgPool2d(kernel_size=7, stride=1)
        self.map1 = nn.Linear(2048 * 2, 512)
        self.map2 = nn.Linear(512, 2048)
        self.fc = nn.Linear(2048, num_classes)
        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, targets=None, flag='train'):
        conv_out = self.conv(images)
        # 输入特征图尺寸 [B, 2048, 7, 7]，平均池化后输出 [B, 2048, 1, 1]，view后变为 [B, 2048]
        pool_out = self.avg(conv_out).view(conv_out.size(0), -1)
        # 如果 targets 为 None，则直接走验证分支
        if flag == 'train' and targets is not None:
            intra_pairs, inter_pairs, intra_labels, inter_labels = self.get_pairs(pool_out, targets)

            features1 = torch.cat([pool_out[intra_pairs[:, 0]], pool_out[inter_pairs[:, 0]]], dim=0)
            features2 = torch.cat([pool_out[intra_pairs[:, 1]], pool_out[inter_pairs[:, 1]]], dim=0)
            labels1 = torch.cat([intra_labels[:, 0], inter_labels[:, 0]], dim=0)
            labels2 = torch.cat([intra_labels[:, 1], inter_labels[:, 1]], dim=0)

            mutual_features = torch.cat([features1, features2], dim=1)
            map1_out = self.map1(mutual_features)
            map2_out = self.drop(map1_out)
            map2_out = self.map2(map2_out)

            gate1 = torch.mul(map2_out, features1)
            gate1 = self.sigmoid(gate1)
            gate2 = torch.mul(map2_out, features2)
            gate2 = self.sigmoid(gate2)

            features1_self = torch.mul(gate1, features1) + features1
            features1_other = torch.mul(gate2, features1) + features1
            features2_self = torch.mul(gate2, features2) + features2
            features2_other = torch.mul(gate1, features2) + features2

            logit1_self = self.fc(self.drop(features1_self))
            logit1_other = self.fc(self.drop(features1_other))
            logit2_self = self.fc(self.drop(features2_self))
            logit2_other = self.fc(self.drop(features2_other))

            return logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2

        else:
            # 当 targets 为 None 或 flag 不是 'train' 时，直接走验证/测试分支
            return self.fc(pool_out)

    def get_pairs(self, embeddings, labels):
        distance_matrix = pdist(embeddings).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy().reshape(-1, 1)
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
        lb_eqs = (labels == labels.T)
        lb_eqs[dia_inds] = False
        dist_same = distance_matrix.copy()
        dist_same[lb_eqs == False] = np.inf
        intra_idxs = np.argmin(dist_same, axis=1)

        dist_diff = distance_matrix.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        inter_idxs = np.argmin(dist_diff, axis=1)

        intra_pairs = np.zeros([embeddings.shape[0], 2])
        inter_pairs = np.zeros([embeddings.shape[0], 2])
        intra_labels = np.zeros([embeddings.shape[0], 2])
        inter_labels = np.zeros([embeddings.shape[0], 2])
        for i in range(embeddings.shape[0]):
            intra_labels[i, 0] = labels[i]
            intra_labels[i, 1] = labels[intra_idxs[i]]
            intra_pairs[i, 0] = i
            intra_pairs[i, 1] = intra_idxs[i]

            inter_labels[i, 0] = labels[i]
            inter_labels[i, 1] = labels[inter_idxs[i]]
            inter_pairs[i, 0] = i
            inter_pairs[i, 1] = inter_idxs[i]

        intra_labels = torch.from_numpy(intra_labels).long().to(device)
        intra_pairs = torch.from_numpy(intra_pairs).long().to(device)
        inter_labels = torch.from_numpy(inter_labels).long().to(device)
        inter_pairs = torch.from_numpy(inter_pairs).long().to(device)

        return intra_pairs, inter_pairs, intra_labels, inter_labels

if __name__ == '__main__':
    # 构造 API_Net 时传入 num_classes 参数，例如 13
    img = torch.randn(1, 3, 224, 224)  # 注意这里使用 224x224，因为数据增强设置如此
    apinet = API_Net(num_classes=13)
    # 验证模式下，输出应为 [batch, 13]，此处输出形状应为 torch.Size([1, 13])
    fea = apinet(img, flag='val')
    print(fea.shape)
