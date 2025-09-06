import random
import torch
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn as nn
from datetime import datetime
import os

from torchvision.models import AlexNet
from BCNN import BCNN
from CMAL import build_network_wrapper
from models import Vgg, ResNet50, Alex_Net
from APINET import API_Net
from Cross import resnet34_osme

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "./"))

# 配置参数
class Config:
    data_root = os.path.join(root_dir, 'Img')  # 图片根目录
    train_list = os.path.join(root_dir, 'Anno_fine', "train.txt")  # 训练集图片路径
    test_list = os.path.join(root_dir, 'Anno_fine', "test.txt")  # 测试集图片路径
    val_list = os.path.join(root_dir, 'Anno_fine', "val.txt")  # 验证集图片路径
    train_labels = os.path.join(root_dir, 'Anno_fine', "train_cate.txt")  # 训练集标签
    test_labels = os.path.join(root_dir, 'Anno_fine', "test_cate.txt")  # 测试集标签
    val_labels = os.path.join(root_dir, 'Anno_fine', "val_cate.txt")  # 验证集标签
    batch_size = 32
    num_workers = 4
    img_size = 448
    lr = 0.001
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取类别索引，并自动修正标签范围
def get_label_mapping(label_file):
    with open(label_file, "r") as f:
        unique_labels = sorted(set(int(line.strip()) for line in f))  # 获取所有独立的类别
    # 生成类别映射表
    label_map = {old_label: new_idx for new_idx, old_label in enumerate(unique_labels)}
    return label_map, len(unique_labels)  # 返回映射表和类别数

# 获取 train、test、val 共有的类别映射
label_map, Config.num_classes = get_label_mapping(Config.train_labels)

# 服饰分类数据集
class DeepFashionDataset(Dataset):
    def __init__(self, img_list, label_file, transform=None):
        with open(img_list, 'r') as f:
            self.img_names = [line.strip() for line in f]
        with open(label_file, 'r') as f:
            self.labels = [label_map[int(line.strip())] for line in f]  # 自动映射标签
        assert len(self.img_names) == len(self.labels), "图像数量和标签数量不匹配！"
        self.transform = transform
        self.root = Config.data_root

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据增强
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
train_dataset = DeepFashionDataset(Config.train_list, Config.train_labels, train_transform)
val_dataset = DeepFashionDataset(Config.val_list, Config.val_labels, val_transform)
test_dataset = DeepFashionDataset(Config.test_list, Config.test_labels, val_transform)

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

# 评估函数
def evaluate(model, test_loader):
    model.eval()
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for img, lab in test_loader:
            img, lab = img.to(Config.device), lab.to(Config.device)
            outputs = model(img)
            # 如果返回的是tuple，则提取聚合分类输出（此处假设为索引2）
            if isinstance(outputs, tuple):
                outputs = outputs[2]
            _, top1_pred = torch.topk(outputs, k=1, dim=1)
            _, top3_pred = torch.topk(outputs, k=3, dim=1)
            _, top5_pred = torch.topk(outputs, k=5, dim=1)
            lab_expanded = lab.view(-1, 1)
            top1_correct += torch.sum(top1_pred == lab_expanded).item()
            top3_correct += torch.sum(top3_pred == lab_expanded).item()
            top5_correct += torch.sum(top5_pred == lab_expanded).item()
            total += lab.size(0)

    top1_acc = 100 * top1_correct / total
    top3_acc = 100 * top3_correct / total
    top5_acc = 100 * top5_correct / total
    return top1_acc, top3_acc, top5_acc

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

net_maker = {
    'ResNet50': ResNet50,
    'AlexNet' : Alex_Net,
    'Vgg' : Vgg,
    'BCNN': BCNN,
    'API-NET' : API_Net,
    'ResNet_OSME': resnet34_osme,
    'CMAL' : build_network_wrapper,
}

if __name__ == "__main__":
    set_seed(42)

    # 选择数据集和模型
    print('-' * 30)
    dataset_config = {0: "DeepFashion"}
    print(dataset_config)
    dataset_name = dataset_config[int(input("Which dataset needs to be trained?\n"))]
    print('-' * 30)
    model_config = {1: "ResNet50", 2: "AlexNet", 3: "Vgg", 4: "BCNN", 5: "API-NET", 6: "ResNet_OSME", 7: "CMAL"}
    print(model_config)
    model_name = model_config[int(input("Which model needs to be trained?\n"))]
    print('-' * 30)

    # 初始化模型
    net = net_maker[model_name](num_classes=Config.num_classes).to(Config.device)
    net = torch.nn.DataParallel(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=Config.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=160, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    now = datetime.now()

    # 初始化最佳指标跟踪
    best_metrics = {
        'top1': {'acc': 0, 'epoch': 0},
        'top3': {'acc': 0, 'epoch': 0},
        'top5': {'acc': 0, 'epoch': 0}
    }

    # 模型保存路径
    model_save_dir = os.path.join(root_dir, "pretrained", model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    for epoch in range(Config.epochs):
        # 评估阶段
        top1_acc, top3_acc, top5_acc = evaluate(model=net, test_loader=test_loader)

        # 更新最佳指标
        current_metrics = {
            'top1': top1_acc,
            'top3': top3_acc,
            'top5': top5_acc
        }

        # 保存最佳模型
        for metric in best_metrics:
            if current_metrics[metric] > best_metrics[metric]['acc']:
                best_metrics[metric]['acc'] = current_metrics[metric]
                best_metrics[metric]['epoch'] = epoch

                # 保存对应指标的最佳模型
                model_path = os.path.join(
                    model_save_dir,
                    f"best_{metric}_epoch{epoch}_acc{current_metrics[metric]:.2f}.pth"
                )
                torch.save(net.state_dict(), model_path)

        # 打印当前epoch结果
        print(f"\nEpoch {epoch} Results:")
        print(f"Top-1 Acc: {top1_acc:.2f}% | Top-3 Acc: {top3_acc:.2f}% | Top-5 Acc: {top5_acc:.2f}%")
        print("Best So Far:")
        for metric in best_metrics:
            print(f"{metric.upper()}: {best_metrics[metric]['acc']:.2f}% @ epoch {best_metrics[metric]['epoch']}")

        # 训练阶段
        net.train()
        for batch_cnt, (image, label) in enumerate(train_loader):
            image, label = image.to(Config.device), label.to(Config.device).long()  # 确保 label 为 long

            optimizer.zero_grad()
            logits = net(image)
            # 如果返回的是tuple，则提取聚合分类输出（此处假设为索引2）
            if isinstance(logits, tuple):
                logits = logits[2]
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            if batch_cnt % 10 == 0:
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                print(f"Epoch {epoch}, Iter {batch_cnt}, Loss: {loss.item():.4f}")

        scheduler.step()

    # 最终测试
    final_top1, final_top3, final_top5 = evaluate(model=net, test_loader=test_loader)
    print(f"\nFinal Test Performance:")
    print(f"Top-1: {final_top1:.2f}% | Top-3: {final_top3:.2f}% | Top-5: {final_top5:.2f}%")
