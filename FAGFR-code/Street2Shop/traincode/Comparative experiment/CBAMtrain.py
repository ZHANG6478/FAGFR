import random
import torch
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import pandas as pd
import numpy as np
import torch.nn as nn
from datetime import datetime
import os

from tqdm import tqdm
from newCBAMmodel import ResNet50WithCBAM
from models import ResNet50, Vgg, Alex_Net

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "./"))


def process_parquet_files(data_type):
    if data_type == 'train':
        file_prefix = 'train-'
        total_files = 165
        data_folder = 'TrainData'
    elif data_type == 'test':
        file_prefix = 'test-'
        total_files = 42
        data_folder = 'TestData'
    else:
        raise ValueError(f"Invalid data type: {data_type}. Expected 'train' or 'test'.")

    # 用于存储所有 DataFrame 的列表
    dfs = []

    # 循环读取每个 Parquet 文件
    for i in range(total_files):
        file_name = f"{data_folder}/{file_prefix}{i:05d}-of-{total_files:05d}.parquet"
        try:
            # 读取当前 Parquet 文件
            df = pd.read_parquet(file_name)
            dfs.append(df)
        except FileNotFoundError:
            print(f"文件 {file_name} 未找到，请检查文件路径和文件名。")

    # 合并所有 DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # 用于存储结果的列表
    results = []

    # 遍历合并后的 DataFrame 的每一行
    for index, row in combined_df.iterrows():
        # 获取当前行的 category 数据
        category = row['category']

        # 获取当前行的 street_photo_image 中的图片字节数据
        image_bytes = row['shop_photo_image']['bytes']

        # 将数据以字典形式存入结果列表
        result_dict = {
            'category': category,
            'image_bytes': image_bytes
        }
        results.append(result_dict)

    # 提取所有的 category 值
    categories = [result['category'] for result in results]

    # 去重并排序
    unique_categories = sorted(set(categories))

    # 创建映射字典
    category_to_label = {category: index for index, category in enumerate(unique_categories)}
    print('映射字典：', category_to_label)

    # 更新 results 中的 category 值为标签
    for result in results:
        result['category'] = category_to_label[result['category']]

    return results


# ----------------------------------------Dataset----------------------------------------
class FashionShopDataset(Dataset):
    def __init__(self, num_classes, img_size, results):
        self.data = results
        self.len = len(self.data)
        print('数据集大小：', self.len)
        self.num_classes = num_classes

        # 数据增强，垂直反转，随机水平翻转，防止过拟合
        self.transform = transforms.Compose([
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        result = self.data[idx]
        image_bytes = result['image_bytes']
        label = result['category']

        # 将字节数据转换为 PIL 图像对象
        image = Image.open(io.BytesIO(image_bytes))
        # 统一转换为 RGB 格式，因为有极少数图片是RGBA格式
        image = image.convert('RGB')
        image = self.transform(image)
        return image, label

    def __len__(self):
        return self.len


# loader 数据加载器
def get_loader(num_classes, bs, shuffle, drop_last, workers, results, img_size=448):
    # 创建数据实例
    dataset = FashionShopDataset(num_classes=num_classes, img_size=img_size, results=results)
    # train
    loader = DataLoader(dataset=dataset, num_workers=workers, batch_size=bs, shuffle=shuffle, drop_last=drop_last, pin_memory=True)
    return loader  # 返回加载器供训练、验证或测试使用。

# 设置随机种子以确保实验的可复现性。
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


net_maker = {
    'Resnet50': ResNet50,
    'Vgg': Vgg,
    'Alexnet': Alex_Net,
    'CBAM': ResNet50WithCBAM,
}

# 模型评估函数 evaluate
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    # 获取模型的 num_classes 属性
    if isinstance(model, torch.nn.DataParallel):
        num_classes = model.module.num_classes
    else:
        num_classes = model.num_classes

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for img, lab in test_loader:
            img, lab = img.to(device, non_blocking=True), lab.to(device, non_blocking=True)
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            total += lab.size(0)
            correct += (predicted == lab).sum().item()

            for i in range(lab.size(0)):
                label = lab[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

    acc = 100 * correct / total
    class_acc = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
    return acc, class_acc

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU:{torch.cuda.get_device_name(device)}")
        print(f"Use {device}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU")

    set_seed(42)
    print('-' * 30)
    model_config = {1: "Resnet50", 2: "Vgg", 3: "Alexnet", 4: "CBAM"}
    print(model_config)
    model_name = model_config[int(input("Which model needs to be trained?\n"))]
    print('-' * 30)
    configs = {
        "num_classes": 11,  # ==============这个记得根据映射字典大小修改一下====================
        "img_size": 448,
        "lr": 1e-3,
        "epoch": 50,
        "train": {"workers": 4, "bs": 32, "shuffle": True, "drop_last": False},
        "test": {"workers": 2, "bs": 32, "shuffle": False, "drop_last": False},
        "valid": {"workers": 0, "bs": 32, "shuffle": False, "drop_last": False}
    }

    # 处理训练集 Parquet 文件得到 results
    train_results = process_parquet_files('train')

    train_loader = get_loader(
        num_classes=configs["num_classes"],
        bs=configs["train"]["bs"],
        shuffle=configs["train"]["shuffle"],
        drop_last=configs["train"]["drop_last"],
        workers=configs["train"]["workers"],
        results=train_results,
        img_size=configs["img_size"]
    )

    # 处理测试集 Parquet 文件得到 test_results
    test_results = process_parquet_files('test')

    test_loader = get_loader(
        num_classes=configs["num_classes"],
        bs=configs["test"]["bs"],
        shuffle=configs["test"]["shuffle"],
        drop_last=configs["test"]["drop_last"],
        workers=configs["test"]["workers"],
        results=test_results,
        img_size=configs["img_size"]
    )

    # 加载模型
    net = net_maker[model_name](num_classes=configs["num_classes"]).to(device)
    net = torch.nn.DataParallel(net).to(device)

    model_path = os.path.join(root_dir, "save", "model", model_name, "epoch_44_acc_89.696.pth")
    net.load_state_dict(torch.load(model_path, weights_only=True))

    # 评估模型
    test_accuracy, class_accuracy = evaluate(net, test_loader)
    print(f"Test Accuracy: {test_accuracy}%")
    print("每个小类的准确率:")
    for i, acc in enumerate(class_accuracy):
        print(f"类别 {i}: {acc}%")
