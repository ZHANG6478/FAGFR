import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import argparse
from datetime import datetime

# 导入各个模型
from BCNN import BCNN
from CMAL import build_network_wrapper
from Cross import resnet34_osme
from models import Vgg, ResNet50, Alex_Net
from APINET import API_Net

# 获取当前脚本所在目录作为根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "./"))

#############################
# 1. 配置 (Config)
#############################
class Config:
    data_root = os.path.join(root_dir, 'Img')  # 图片根目录
    train_list = os.path.join(root_dir, 'Anno_fine', "train.txt")
    test_list = os.path.join(root_dir, 'Anno_fine', "test.txt")
    val_list = os.path.join(root_dir, 'Anno_fine', "val.txt")
    train_labels = os.path.join(root_dir, 'Anno_fine', "train_attr.txt")
    test_labels = os.path.join(root_dir, 'Anno_fine', "test_attr.txt")
    val_labels = os.path.join(root_dir, 'Anno_fine', "val_attr.txt")

    batch_size = 32
    num_workers = 4
    img_size = 448
    lr = 0.001
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
    # 预训练权重路径已不再使用
    pretrained_model_path = ""

#############################
# 2. 数据集定义
#############################
class FashionAttrDataset(Dataset):
    """
    DeepFashion 数据集（用于属性预测）：
      - list_file：每行存放图片的相对路径
      - attr_file：每行存放 26 维属性标签（0/1），用空格分隔
      行数必须一致。
    """
    def __init__(self, list_file, attr_file, transform=None):
        super().__init__()
        with open(list_file, 'r') as f:
            self.img_paths = [line.strip() for line in f if line.strip()]
        with open(attr_file, 'r') as f:
            self.labels = []
            for line in f:
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
# 3. 模型选择字典
#############################
net_maker = {
    'ResNet50': ResNet50,
    'AlexNet' : Alex_Net,
    'Vgg' : Vgg,
    'BCNN': BCNN,
    'API-NET' : API_Net,
    'ResNet_OSME': resnet34_osme,
    'CMAL' : build_network_wrapper,
}

#############################
# 4. 多标签 Top-k 策略及评估函数
#############################
def multi_label_top_k_global(outputs, targets, group_splits, k=3):
    """
    在全局（26）属性上取 top-k，然后统计:
      - overall：预测的 top-k 与真实标签（>=0.5）有交集，则算命中
      - 每个属性组的命中样本数（使用相同的 top-k 结果）
    """
    prob = torch.sigmoid(outputs)  # (B, 26)
    bsz, nattr = prob.shape
    assert nattr == 26, f"当前仅支持 26 个属性，实际为 {nattr}."
    _, topk_indices = prob.topk(k, dim=1)
    overall_hits = 0
    group_hits = {gname: 0 for gname in group_splits}
    for i in range(bsz):
        true_indices = (targets[i] > 0.5).nonzero(as_tuple=True)[0].tolist()
        pred_indices = topk_indices[i].tolist()
        if set(pred_indices) & set(true_indices):
            overall_hits += 1
        for gname, intervals in group_splits.items():
            group_attr_indices = []
            for (start, end) in intervals:
                group_attr_indices.extend(range(start, end))
            predicted_in_group = set(pred_indices) & set(group_attr_indices)
            true_in_group = set(true_indices) & set(group_attr_indices)
            if predicted_in_group & true_in_group:
                group_hits[gname] += 1
    return overall_hits, group_hits, bsz

def evaluate_model(model, data_loader, group_splits, device, k_values=(1, 3, 5)):
    """
    在验证/测试集上评估模型，计算不同 k 下的整体及各组命中率（单位 %）
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
# 6. 主训练入口 (传入用户选择的模型)
#############################
def main(model):
    set_seed(42)
    device = Config.device
    print("Using device:", device)

    model = model.to(device)
    # 如果模型已经包装为 DataParallel，则无需再次包装
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=Config.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=160, gamma=0.1)

    # 最佳指标记录
    best_overall = {1: 0, 3: 0, 5: 0}
    best_group = {3: {g: 0 for g in Config.group_splits}, 5: {g: 0 for g in Config.group_splits}}
    model_save_dir = os.path.join(root_dir, "pretrained", "SA")
    os.makedirs(model_save_dir, exist_ok=True)

    for epoch in range(Config.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        overall_acc, group_acc = evaluate_model(model, val_loader, Config.group_splits, device, k_values=(1, 3, 5))

        for k in (1, 3, 5):
            if overall_acc[k] > best_overall[k]:
                best_overall[k] = overall_acc[k]
                model_path = os.path.join(model_save_dir, f"best_overall_top{k}_epoch{epoch}_acc{overall_acc[k]:.2f}.pth")
                torch.save(model.state_dict(), model_path)
        for k in (3, 5):
            for g in Config.group_splits:
                if group_acc[k][g] > best_group[k][g]:
                    best_group[k][g] = group_acc[k][g]

        print(f"\nEpoch {epoch} Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print("Validation Overall Accuracies:")
        for k in (1, 3, 5):
            print(f"Top-{k}: {overall_acc[k]:.2f}%")
        print("Validation Group Accuracies:")
        for k in (1, 3, 5):
            for g in Config.group_splits:
                print(f"Top-{k} Group {g}: {group_acc[k][g]:.2f}%")
        print("Best Overall So Far:")
        for k in (1, 3, 5):
            print(f"Top-{k}: {best_overall[k]:.2f}%")
        print("Best Group So Far (Top-3 and Top-5):")
        for k in (3, 5):
            for g in Config.group_splits:
                print(f"Top-{k} Group {g}: {best_group[k][g]:.2f}%")

    final_overall, final_group = evaluate_model(model, test_loader, Config.group_splits, device, k_values=(1, 3, 5))
    print(f"\nFinal Test Performance:")
    for k in (1, 3, 5):
        print(f"Overall Top-{k}: {final_overall[k]:.2f}%")
    for g in Config.group_splits:
        for k in (1, 3, 5):
            print(f"Group {g} Top-{k}: {final_group[k][g]:.2f}%")

# 设置随机种子，保证结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#############################
# 程序入口
#############################
if __name__ == "__main__":
    set_seed(42)

    # 用户选择数据集和模型（当前数据集固定为 DeepFashion）
    print('-' * 30)
    dataset_config = {0: "DeepFashion"}
    print("Available datasets:", dataset_config)
    dataset_name = dataset_config[int(input("Which dataset needs to be trained?\n"))]
    print('-' * 30)
    model_config = {1: "ResNet50", 2: "AlexNet", 3: "Vgg", 4: "BCNN", 5: "API-NET", 6: "CMAL" , 7 :'ResNet_OSME'}
    print("Available models:", model_config)
    model_choice = int(input("Which model needs to be trained?\n"))
    model_name = model_config[model_choice]
    print('-' * 30)

    # 根据选择构造模型
    chosen_model = net_maker[model_name](num_classes=Config.num_classes).to(Config.device)
    # 包装模型为 DataParallel（如果尚未包装）
    if not isinstance(chosen_model, torch.nn.DataParallel):
        chosen_model = torch.nn.DataParallel(chosen_model)

    # 使用命令行参数（此处预留覆盖预训练权重逻辑，但目前不使用）
    parser = argparse.ArgumentParser(description="Train model from scratch")
    parser.add_argument("--pretrained", type=str, default="",
                        help="Path to the pretrained .pth file (unused)")
    args = parser.parse_args()

    # 传入用户选择的模型进行训练
    main(chosen_model)
