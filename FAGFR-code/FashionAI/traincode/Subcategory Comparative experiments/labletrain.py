import random
import torch
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import numpy as np
import torch.nn as nn
from datetime import datetime
import os

from models import ResNet50, Vgg, Alex_Net
from newCBAMmodel import ResNet50WithCBAM

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "./"))


# ----------------------------------------Dataset----------------------------------------
class FashionAIDataset(Dataset):
    def __init__(self, num_classes, data_type, img_size, dataset_name):
        self.img_path = os.path.join(root_dir, 'all_cat', dataset_name, data_type, "image")
        self.label_path = os.path.join(root_dir, 'all_cat', dataset_name, data_type, "label", "label.json")
        self.labels = json_reader(self.label_path)
        self.len = len(self.labels)
        self.num_classes = num_classes

        if data_type == "train":
            self.transform = transforms.Compose([
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        labels = self.labels[idx]
        image_path = os.path.join(self.img_path, labels[0])
        image = Image.open(image_path)
        image = self.transform(image)
        return image, labels[1]

    def __len__(self):
        return self.len


# json读取函数
def json_reader(label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line)
    return json_obj


# loader数据加载器
def get_loader(num_classes, bs, shuffle, drop_last, workers, dataset_type, dataset_name, img_size=448):
    dataset = FashionAIDataset(num_classes=num_classes, data_type=dataset_type, img_size=img_size,
                               dataset_name=dataset_name)
    loader = DataLoader(dataset=dataset, num_workers=workers, batch_size=bs, shuffle=shuffle, drop_last=drop_last)
    return loader


# 原有整体评估函数
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, lab in test_loader:
            img, lab = img.cuda(), lab.cuda()
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            total += lab.size(0)
            correct += (predicted == lab).sum().item()
    acc = 100 * correct / total
    return acc

def evaluate_per_class(model, test_loader, num_classes):
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(labels)):
                # 将标签转换为 0-indexed
                label = labels[i].item() - 1
                class_total[label] += 1
                if predicted[i].item() == labels[i].item():
                    class_correct[label] += 1

    per_class_accuracy = []
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
        else:
            accuracy = 0
        per_class_accuracy.append(accuracy)
    return per_class_accuracy


# 固定随机种子确保复现性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


net_maker = {
    'GroupNet': Net,
    'Resnet50': ResNet50,
    'Vgg': Vgg,
    'Alexnet': Alex_Net,
    'CBAM': ResNet50WithCBAM
}

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,4'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU:{torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU")

    set_seed(42)
    print('-' * 30)
    dataset_config = {0: "coat_length", 1: "collar_design", 2: "lapel_design", 3: "neck_design",
                      4: "neckline_design", 5: "pant_length", 6: "skirt_length",  7: "sleeve_length"}
    print(dataset_config)
    dataset_name = dataset_config[int(input("Which dataset needs to be trained?\n"))]
    print('-' * 30)
    model_config = { 1: "Resnet50", 2: "Vgg", 3: "Alexnet", 4: "CBAM"}
    print(model_config)
    model_name = model_config[int(input("Which model needs to be trained?\n"))]
    print('-' * 30)
    configs = {
        "num_classes": 9,
        "img_size": 448,
        "lr": 1e-2,
        "epoch": 50,
        "train": {"workers": 4, "bs": 64, "shuffle": True, "drop_last": False},
        "test": {"workers": 0, "bs": 32, "shuffle": False, "drop_last": False},
        "valid": {"workers": 0, "bs": 32, "shuffle": False, "drop_last": False}
    }

    train_loader = get_loader(
        num_classes=configs["num_classes"],
        bs=configs["train"]["bs"],
        shuffle=configs["train"]["shuffle"],
        drop_last=configs["train"]["drop_last"],
        workers=configs["train"]["workers"],
        dataset_type="train",
        img_size=configs["img_size"],
        dataset_name=dataset_name
    )

    test_loader = get_loader(
        num_classes=configs["num_classes"],
        bs=configs["test"]["bs"],
        shuffle=configs["test"]["shuffle"],
        drop_last=configs["test"]["drop_last"],
        workers=configs["test"]["workers"],
        dataset_type="test",
        img_size=configs["img_size"],
        dataset_name=dataset_name
    )

    # 选择模型，假设使用的是 num_classes=13 的网络（注意：实际可能需要根据具体数据集和任务修改）
    net = net_maker[model_name](num_classes=configs["num_classes"]).cuda()
    net = torch.nn.DataParallel(net)

    lr, momentum, decay_step, weight_decay = configs["lr"], 0.9, 160, 1e-5
    optimizer = torch.optim.SGD(lr=lr, momentum=momentum, weight_decay=weight_decay, params=net.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    now = datetime.now()

    # 定义一个列表记录细粒度类别（例如 coat_length 有 7 个类别）的最佳准确率
    #-----------------------------!!!!!!!!!!!!!!!111------------------
    num_fine_classes = 9
    best_per_class = [0.0] * num_fine_classes
    acc_best = {"acc": 0, "epoch": 0}

    for epoch in range(configs["epoch"]):
        # 每个 epoch 进行一次验证，计算整体和各类别准确率
        if epoch % 1 == 0:
            overall_acc = evaluate(model=net, test_loader=test_loader)
            per_class_acc = evaluate_per_class(model=net, test_loader=test_loader, num_classes=num_fine_classes)

            # 更新每个细类别的最佳准确率
            for i in range(num_fine_classes):
                best_per_class[i] = max(best_per_class[i], per_class_acc[i])

            print("Epoch {}: Overall Accuracy: {:.2f}%".format(epoch, overall_acc))
            for idx, acc in enumerate(per_class_acc):
                print("Label {} Accuracy: {:.2f}%".format(idx + 1, acc))

            if overall_acc > acc_best["acc"]:
                acc_best["acc"] = overall_acc
                acc_best["epoch"] = epoch
                model_path = os.path.join(root_dir, "save", "model", model_name,
                                          "epoch_" + str(epoch) + "_acc_" + str(round(overall_acc, 3)) + ".pth")
                torch.save(net.state_dict(), model_path)
            print("Best overall acc so far: {:.2f}% at epoch {}".format(acc_best["acc"], acc_best["epoch"]))

        net.train(True)
        current_lr = optimizer.param_groups[0]['lr']
        print("Epoch: {}  Current learning rate: {:.5f}".format(epoch, current_lr))
        for batch_cnt, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), torch.LongTensor(labels).cuda()
            optimizer.zero_grad()
            logits = net(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if batch_cnt % 10 == 0:
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                print("Epoch {} Iteration {}: Loss: {:.4f}".format(epoch, batch_cnt, loss.item()))
        scheduler.step()

    # 训练结束后输出每个细粒度标签的最佳准确率（标签从 1 到 7）
    print("\n训练结束，各细粒度标签最佳准确率：")
    for idx, acc in enumerate(best_per_class):
        print("Label {} best acc: {:.2f}%".format(idx + 1, acc))
