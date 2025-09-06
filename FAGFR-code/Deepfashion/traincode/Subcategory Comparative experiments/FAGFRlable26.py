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
import argparse
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
    epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 26  # 26 个属性
    # 属性组配置（若不需要分组评估，可忽略此配置）
    group_splits = {
        'Texture': [(0, 7)],
        'Shape': [(10, 13)],
        'Fabric': [(17, 23)],
        'Style': [(23, 26)],
        'Part': [(7, 10), (13, 17)]
    }
    # 默认预训练权重文件路径
    pretrained_model_path = os.path.join(root_dir, "pretrained", "model_group_api_lsk_encoder_acc_88.664.pth")

# 定义26个属性的名称（顺序需与标签文件中的顺序对应）
attribute_names = [
    "floral", "graphic", "striped", "embroidered", "pleated", "solid", "lattice",
    "long_sleeve", "short_sleeve", "sleeveless", "maxi_length", "mini_length", "no_dress",
    "crew_neckline", "v_neckline", "square_neckline", "no_neckline", "denim", "chiffon",
    "cotton", "leather", "faux", "knit", "tight", "loose", "conventional"
]

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
# 3. 模型定义（使用 model_group_api_lsk_encoder 中的 Net）
#############################
def build_net():
    # 设置 num_groups 为 32, hidden_c 为 256
    return ResNet50WithCBAM(num_classes=Config.num_classes, num_groups=32, hidden_c=256)

# 设置随机种子，确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#############################
# 4. 评估函数：计算每个属性在 top-3 和 top-5 下的正确率
#############################
def evaluate_attribute_accuracy(model, data_loader, device, k_values=(3, 5)):
    """
    计算每个属性的 top-k 正确率（仅针对正样本）。
    对于每个样本：
      - 若某属性的真实标签为1，则统计其是否在该样本的 top-k 预测中。
    返回一个字典：{k: [acc_attr0, acc_attr1, ..., acc_attr25]}
    """
    num_attributes = Config.num_classes  # 26
    # 初始化每个 k 下的正样本总数与命中数
    pos_counts = {k: np.zeros(num_attributes) for k in k_values}
    correct_counts = {k: np.zeros(num_attributes) for k in k_values}

    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)  # shape: (B, 26)
            outputs = model(images)  # shape: (B, 26)
            probs = torch.sigmoid(outputs)  # 转为概率
            batch_size = labels.size(0)
            for k in k_values:
                # 对每个样本取 top-k 的预测索引
                _, topk_indices = probs.topk(k, dim=1)  # shape: (B, k)
                for i in range(batch_size):
                    # 获取当前样本中正样本的属性索引（值大于0.5视为正）
                    true_indices = (labels[i] > 0.5).nonzero(as_tuple=True)[0]
                    topk_pred = topk_indices[i].tolist()
                    for attr in true_indices:
                        attr_idx = attr.item()
                        pos_counts[k][attr_idx] += 1
                        if attr_idx in topk_pred:
                            correct_counts[k][attr_idx] += 1
    # 计算每个属性的准确率（避免除0）
    accuracies = {}
    for k in k_values:
        acc_list = []
        for i in range(num_attributes):
            if pos_counts[k][i] > 0:
                acc = 100 * correct_counts[k][i] / pos_counts[k][i]
            else:
                acc = 0.0
            acc_list.append(acc)
        accuracies[k] = acc_list
    return accuracies

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

    # 实例化模型
    net = build_net().to(device)

    # 如果存在预训练模型文件，则加载预训练权重
    if os.path.exists(Config.pretrained_model_path):
        print(f"Loading pretrained weights from {Config.pretrained_model_path}")
        state_dict = torch.load(Config.pretrained_model_path, map_location=device)
        # 如果保存权重时使用了 DataParallel，则去除 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v
        net.load_state_dict(new_state_dict, strict=False)
    else:
        print("No pretrained weights found, training from scratch.")

    net = torch.nn.DataParallel(net)
    nn.Module.load_state_dict = original_load_state_dict

    # 定义损失函数、优化器和学习率调度器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=Config.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=160, gamma=0.1)

    # 用于跟踪验证集上每个属性的最佳准确率（分别对 top-3 和 top-5）
    best_attr_acc = {3: [0.0] * Config.num_classes, 5: [0.0] * Config.num_classes}

    # 训练过程，每个 epoch 结束后输出26个标签的 top-3 与 top-5 正确率，并输出至今最佳的各属性正确率
    for epoch in range(Config.epochs):
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_loss = validate_one_epoch(net, val_loader, criterion, device)
        print(f"Epoch {epoch} Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # 计算当前验证集上每个属性的 Top-3 与 Top-5 正确率
        val_attr_acc = evaluate_attribute_accuracy(net, val_loader, device, k_values=(3, 5))
        for k in (3, 5):
            print(f"Epoch {epoch} Current Top-{k} Accuracy:")
            for i, attr in enumerate(attribute_names):
                print(f"  {attr}: {val_attr_acc[k][i]:.2f}%")
            # 更新最佳记录
            for i in range(Config.num_classes):
                if val_attr_acc[k][i] > best_attr_acc[k][i]:
                    best_attr_acc[k][i] = val_attr_acc[k][i]

        # 输出至今最佳的各属性正确率
        print(f"Epoch {epoch} Best So Far:")
        for k in (3, 5):
            print(f"Best Top-{k} Accuracy:")
            for i, attr in enumerate(attribute_names):
                print(f"  {attr}: {best_attr_acc[k][i]:.2f}%")
        print("-" * 60)

    # 最后在测试集上计算并输出每个属性的 Top-3 与 Top-5 正确率
    final_attr_acc = evaluate_attribute_accuracy(net, test_loader, device, k_values=(3, 5))
    print(f"\nFinal Test Per-Attribute Accuracies:")
    for k in (3, 5):
        print(f"Top-{k} Accuracy:")
        for i, attr in enumerate(attribute_names):
            print(f"  {attr}: {final_attr_acc[k][i]:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with pretrained weights")
    parser.add_argument("--pretrained", type=str, default="",
                        help="Path to the pretrained .pth file")
    args = parser.parse_args()
    if args.pretrained:
        Config.pretrained_model_path = args.pretrained
    main()
