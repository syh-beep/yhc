import os
import sys
import json
import pickle
import random

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]

    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        plt.xticks(range(len(flower_class)), flower_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def read_split_data_from_folders(train_root: str, val_root: str):
    assert os.path.exists(train_root), f"train root {train_root} does not exist."
    assert os.path.exists(val_root), f"val root {val_root} does not exist."

    classes = [cla for cla in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, cla))]
    classes.sort()
    class_indices = dict((k, v) for v, k in enumerate(classes))

    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]

    for cla in classes:
        cla_path = os.path.join(train_root, cla)
        if not os.path.isdir(cla_path):
            continue
        images = [os.path.join(cla_path, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        label = class_indices[cla]
        train_images_path.extend(images)
        train_images_label.extend([label] * len(images))

    for cla in classes:
        cla_path = os.path.join(val_root, cla)
        if not os.path.isdir(cla_path):
            print(f"Warning: {cla_path} not found, skip.")
            continue
        images = [os.path.join(cla_path, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        label = class_indices[cla]
        val_images_path.extend(images)
        val_images_label.extend([label] * len(images))

    print(f"Total images: {len(train_images_path) + len(val_images_path)}")
    print(f"Training images: {len(train_images_path)}")
    print(f"Validation images: {len(val_images_path)}")

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    训练一个epoch，返回平均损失和准确率
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        pred = model(images)
        loss = loss_function(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred_idx = torch.max(pred, dim=1)
        total_correct += (pred_idx == labels).sum().item()
        total_samples += labels.size(0)

        mean_loss = total_loss / (step + 1)
        acc = total_correct / total_samples
        data_loader.desc = f"[epoch {epoch}] mean loss: {mean_loss:.4f}, acc: {acc:.4f}"

    avg_loss = total_loss / len(data_loader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    验证集评估，返回平均损失、准确率以及所有预测标签和真实标签（用于后续计算混淆矩阵）
    """
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        pred = model(images)
        loss = loss_function(pred, labels)

        total_loss += loss.item()
        _, pred_idx = torch.max(pred, dim=1)
        total_correct += (pred_idx == labels).sum().item()
        total_samples += labels.size(0)

        all_preds.extend(pred_idx.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc, np.array(all_preds), np.array(all_labels)


def calculate_metrics(all_labels, all_preds, class_names):
    """
    计算每个类别的精确率、召回率、特异度、F1分数以及混淆矩阵
    返回字典和混淆矩阵
    """
    cm = confusion_matrix(all_labels, all_preds)
    num_classes = len(class_names)
    metrics = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[class_names[i]] = {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1
        }
    return metrics, cm