#细粒度分类的柱状图
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
from CBAM import ResNet50WithCBAM

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "./"))


# ----------------------------------------Dataset----------------------------------------
class FashionAIDataset(Dataset):
    def __init__(self, num_classes, data_type, img_size, dataset_name):
        self.img_path = os.path.join(root_dir, 'all_cat', dataset_name, data_type, "image")
        self.label_path = os.path.join(root_dir, 'all_cat', dataset_name, data_type, "label", "label.json")
        #print(self.img_path)
        #print(self.label_path)
        self.labels = json_reader(self.label_path)
        self.len = len(self.labels)
        self.num_classes = num_classes

        if data_type == "train":
            #数据增强，垂直反转，随机水平翻转，防止过拟合
            self.transform = transforms.Compose([
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            #验证集不进行数据增强
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

#json读取函数
def json_reader(label_path):
    # 打开JSON文件，逐行读取并解析
    # label = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line)
            # print(json_obj)
            # label.append(json_obj)
    # return label
    return json_obj

#loader数据加载器
def get_loader(num_classes, bs, shuffle, drop_last, workers, dataset_type, dataset_name, img_size=448):
    #创建数据实例
    dataset = FashionAIDataset(num_classes=num_classes, data_type=dataset_type, img_size=img_size, dataset_name=dataset_name)
    # train
    loader = DataLoader(dataset=dataset, num_workers=workers, batch_size=bs, shuffle=shuffle, drop_last=drop_last)
    return loader#返回加载器供训练、验证或测试使用。

#模型评估函数 evaluate
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    for img, lab in test_loader:
        img, lab = img.cuda(), lab.cuda()
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        total += lab.size(0)
        # print(f"label.shape:{lab.shape}, logits.shape={logits.shape}")
        correct += (predicted == lab).sum().item()
        # print(f"total = {total}, correct = {correct}")
    acc = 100 * correct / total
    return acc

#设置随机种子以确保实验的可复现性。
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

net_maker = {
    'Resnet50':ResNet50,
    'Vgg' : Vgg,
    'Alexnet':Alex_Net,
    'CBAM': ResNet50WithCBAM,
}


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU:{torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU")

    set_seed(42)
    print('-' * 30)
    dataset_config = {0: "coat_length",1:"collar_design",2:"lapel_design",3:"neck_design",4:"neckline_design"
                      ,5:"pant_length" , 6:"skirt_length",7:"skirt_length",8:"sleeve_length"}
    print(dataset_config)
    dataset_name = dataset_config[int(input("Which dataset needs to trained?\n"))]
    print('-' * 30)
    model_config = { 1: "Resnet50", 2: "Vgg" , 3: "Alexnet", 4: "CBAM" }
    print(model_config)
    model_name = model_config[int(input("Which model needs to be trained?\n"))]
    print('-' * 30)
    configs = {
        "num_classes": 9,
        "img_size": 448,
        "lr": 1e-2,
        "epoch": 50,
        "train": {"workers": 4, "bs": 32, "shuffle": True, "drop_last": False},
        "test": {"workers": 0, "bs": 32, "shuffle": False, "drop_last": False},
        "valid": {"workers": 0, "bs": 32, "shuffle": False, "drop_last": False}
    }
    print(configs["num_classes"])
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

    # 3. Resnet101
    net = net_maker[model_name](num_classes=configs["num_classes"]).cuda()
    net = torch.nn.DataParallel(net)

    lr, momentum, decay_step, weight_decay = configs["lr"], 0.9, 40, 1e-5
    optimizer = torch.optim.SGD(lr=lr, momentum=momentum, weight_decay=weight_decay, params=net.parameters())

    scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    now = datetime.now()

    acc_best = {"acc": 0, "epoch": 0}
    for epoch in range(0, configs["epoch"], 1):
        if epoch % 1 == 0:
            accuracy = evaluate(model=net, test_loader=test_loader)
            if accuracy > acc_best["acc"]:
                acc_best["acc"] = accuracy
                acc_best["epoch"] = epoch
                model_path = os.path.join(
                    root_dir, "save", "model", model_name,
                    "epoch_" +  str(epoch) + "_acc_" + str(round(accuracy, 3)) + ".pth")
                torch.save(net.state_dict(), model_path)
            print(" ---------------- ACC ----------------")
            print(f"The best is:{acc_best}, appear in epoch:{acc_best['epoch']}")

        net.train(True)
        current_lr = optimizer.param_groups[0]['lr']
        print("Epoch: ", epoch, "Current learning rate: ", current_lr)
        for batch_cnt, batch in enumerate(train_loader):
            image, label = batch[0].cuda(), torch.LongTensor(batch[1]).cuda()
            optimizer.zero_grad()
            # print(f"学习率：{optimizer.param_groups[0]['lr']}")
            logits = net(image)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            # if epoch % decay_step == 0 and epoch != 0:
            if batch_cnt % 10 == 0:
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                print(f"epoch = {epoch}, iteration: {batch_cnt},loss: {loss.item()}")
        scheduler.step()