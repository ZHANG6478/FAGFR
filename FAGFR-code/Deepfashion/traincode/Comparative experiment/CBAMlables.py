import datetime
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mpmath.calculus.extrapolation import nprod  # 如果不使用可移除
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# 假设 model_sa.py 中定义了 Net（请确保该文件可用）
#from model_sa import Net
from models import ResNet50
from newCBAMmodel import ResNet50WithCBAM

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "./"))


#############################
# 1. 配置 (Config)
#############################
class Config:
    data_root = os.path.join(root_dir, 'Img')  # 图片根目录
    # 注意：这里使用属性标签文件
    train_list = os.path.join(root_dir, 'Anno_fine', "train.txt")
    test_list = os.path.join(root_dir, 'Anno_fine', "test.txt")
    val_list = os.path.join(root_dir, 'Anno_fine', "val.txt")
    train_labels = os.path.join(root_dir, 'Anno_fine', "train_attr.txt")
    test_labels = os.path.join(root_dir, 'Anno_fine', "test_attr.txt")
    val_labels = os.path.join(root_dir, 'Anno_fine', "val_attr.txt")

    batch_size = 32
    num_workers = 4
    img_size = 448
    lr = 0.001  # 学习率
    epochs = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 26  # 26 个属性
    # 属性组配置（各组对应的属性索引范围）
    group_splits = {
        'Texture': [(0, 7)],
        'Shape': [(10, 13)],
        'Fabric': [(17, 23)],
        'Style': [(23, 26)],
        'Part': [(7, 10), (13, 17)]
    }


#############################
# 2. 数据集定义
#############################
class FashionAttrDataset(Dataset):
    """
    用于属性预测的 DeepFashion 数据集：
      - list_file：每行存放图片的相对路径
      - attr_file：每行存放 26 维属性标签（0/1），用空格分隔
      二者行数应一致。
    """

    def __init__(self, list_file, attr_file, transform=None):
        super().__init__()
        with open(list_file, 'r') as f:
            self.img_paths = [line.strip() for line in f if line.strip()]
        with open(attr_file, 'r') as f:
            self.labels = []
            for line in f:
                # 将每行分割为 26 个浮点数
                arr = [float(x) for x in line.strip().split()]
                self.labels.append(arr)
        assert len(self.img_paths) == len(self.labels), "图片数和属性数不一致"
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(Config.data_root, self.img_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # 26 维向量
        if self.transform:
            image = self.transform(image)
        return image, label


# 数据增强与预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = FashionAttrDataset(Config.train_list, Config.train_labels, train_transform)
val_dataset = FashionAttrDataset(Config.val_list, Config.val_labels, val_transform)
test_dataset = FashionAttrDataset(Config.test_list, Config.test_labels, val_transform)

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)


#############################
# 3. 模型定义（使用 model_sa.py 中的 Net）
#############################
def build_net():
    # 直接实例化 Net 类，传入类别数、分组数和隐藏通道数
    # return Net(num_classes=Config.num_classes, num_groups=5, hidden_c=1024)
    # return  ResNet50(num_classes=Config.num_classes)
    return ResNet50WithCBAM(num_classes=Config.num_classes,num_groups=5)
# 设置随 机种子，确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#############################
# 4. 多标签 Top-k 策略及评估函数
#############################
def multi_label_top_k_global(outputs, targets, group_splits, k=3):
    """
    在全局(26)属性上先取 top-k，然后分别统计:
      - overall 命中样本数：若预测的 top-k 与真实标签 (>=0.5) 有交集，则命中
      - 每个 group 的命中样本数（用相同的全局 top-k）
    参数:
      outputs: (B,26) - 模型输出 logits
      targets: (B,26) - 真实标签 (0/1)
      group_splits: dict，如 {"Group1": [(0,7)], ...}，指示各组的属性索引区间
      k: 取 top-k
    返回:
      overall_hits: 整个 batch 中在 top-k 与真实标签有交集的样本数
      group_hits: { group_name: 命中样本数 }
      total_samples: B
    """
    prob = torch.sigmoid(outputs)  # (B,26)
    bsz, nattr = prob.shape
    assert nattr == 26, f"本例只考虑 26 个属性，这里是 {nattr}."
    _, topk_indices = prob.topk(k, dim=1)
    overall_hits = 0
    group_hits = {gname: 0 for gname in group_splits}
    for i in range(bsz):
        true_indices = (targets[i] > 0.5).nonzero(as_tuple=True)[0].tolist()
        pred_indices = topk_indices[i].tolist()
        if len(set(pred_indices) & set(true_indices)) > 0:
            overall_hits += 1
        for gname, intervals in group_splits.items():
            group_attr_indices = []
            for (start, end) in intervals:
                group_attr_indices.extend(range(start, end))
            predicted_in_group = set(pred_indices) & set(group_attr_indices)
            true_in_group = set(true_indices) & set(group_attr_indices)
            if len(predicted_in_group & true_in_group) > 0:
                group_hits[gname] += 1
    return overall_hits, group_hits, bsz


def evaluate_model(model, data_loader, group_splits, device, k_values=(1, 3, 5)):
    """
    在验证/测试数据集上评估模型，计算不同 k 下的整体和各组命中率（单位：%）。
    返回:
      overall_acc: {k: overall_accuracy}
      group_acc: {k: {group_name: accuracy}}
    """
    model.eval()
    total_samples = 0
    hits = {k: 0 for k in k_values}
    group_hits = {k: {g: 0 for g in group_splits} for k in k_values}
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            bs = images.size(0)
            total_samples += bs
            for k in k_values:
                overall, grp, _ = multi_label_top_k_global(outputs, labels, group_splits, k=k)
                hits[k] += overall
                for g in group_splits:
                    group_hits[k][g] += grp[g]
    overall_acc = {k: 100 * hits[k] / total_samples for k in k_values}
    group_acc = {k: {g: 100 * group_hits[k][g] / total_samples for g in group_splits} for k in k_values}
    return overall_acc, group_acc


#############################
# 5. 训练和验证流程
#############################
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        # 多标签任务使用 BCEWithLogitsLoss，标签保持 float 类型
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
    return total_loss / total_samples


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
    return total_loss / total_samples


#############################
# 6. 主训练入口
#############################
def main():
    set_seed(42)
    device = Config.device
    print("Using device:", device)

    # ===== 在实例化模型前打补丁（用于加载预训练权重时忽略尺寸不匹配） =====
    original_load_state_dict = nn.Module.load_state_dict

    def patched_load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        for key in list(state_dict.keys()):
            if key in model_dict and state_dict[key].shape != model_dict[key].shape:
                print(f"Removing key {key} from checkpoint because shape mismatch: "
                      f"checkpoint shape {state_dict[key].shape}, model shape {model_dict[key].shape}")
                del state_dict[key]
        return original_load_state_dict(self, state_dict, strict=False)

    nn.Module.load_state_dict = patched_load_state_dict

    # 实例化模型，此时 Net 内部加载预训练权重时会使用补丁
    net = build_net().to(device)
    net = torch.nn.DataParallel(net)
    # 恢复原 load_state_dict 方法
    nn.Module.load_state_dict = original_load_state_dict

    # 使用 BCEWithLogitsLoss 作为损失函数（多标签任务）
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=Config.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=160, gamma=0.1)

    # 用于记录整体最佳指标（以 Top-1/Top-3/Top-5 为准）
    best_overall = {1: 0, 3: 0, 5: 0}
    # 新增：用于记录各组在 Top-3 和 Top-5 下的最佳准确率
    best_group = {
        3: {group: 0 for group in Config.group_splits},
        5: {group: 0 for group in Config.group_splits}
    }
    model_save_dir = os.path.join(root_dir, "pretrained", "SA")
    os.makedirs(model_save_dir, exist_ok=True)

    for epoch in range(Config.epochs):
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_loss = validate_one_epoch(net, val_loader, criterion, device)
        overall_acc, group_acc = evaluate_model(net, val_loader, Config.group_splits, device, k_values=(1, 3, 5))

        # 更新整体最佳准确率
        for k in (1, 3, 5):
            if overall_acc[k] > best_overall[k]:
                best_overall[k] = overall_acc[k]
                # 保存整体指标最佳模型（例如保存 Top-k 最佳模型）
                model_path = os.path.join(model_save_dir,
                                          f"best_overall_top{k}_epoch{epoch}_acc{overall_acc[k]:.2f}.pth")
                torch.save(net.state_dict(), model_path)

        # 更新各组最佳准确率（仅更新 Top-3 和 Top-5）
        for k in (3, 5):
            for group in Config.group_splits:
                if group_acc[k][group] > best_group[k][group]:
                    best_group[k][group] = group_acc[k][group]

        print(f"\nEpoch {epoch} Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print("Validation Overall Accuracies:")
        for k in (1, 3, 5):
            print(f"Top-{k}: {overall_acc[k]:.2f}%")
        print("Validation Group Accuracies:")
        for k in (1, 3, 5):
            for group in Config.group_splits:
                print(f"Top-{k} Group {group}: {group_acc[k][group]:.2f}%")
        print("Best Overall Accuracies So Far:")
        for k in (1, 3, 5):
            print(f"Top-{k}: {best_overall[k]:.2f}%")
        print("Best Group Accuracies So Far (Only Top-3 and Top-5):")
        for k in (3, 5):
            for group in Config.group_splits:
                print(f"Top-{k} Group {group}: {best_group[k][group]:.2f}%")

    final_overall, final_group = evaluate_model(net, test_loader, Config.group_splits, device, k_values=(1, 3, 5))
    print(f"\nFinal Test Performance:")
    for k in (1, 3, 5):
        print(f"Overall Top-{k}: {final_overall[k]:.2f}%")
    for group in Config.group_splits:
        for k in (1, 3, 5):
            print(f"Group {group} Top-{k}: {final_group[k][group]:.2f}%")


if __name__ == "__main__":
    main()
