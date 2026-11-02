# train_improved.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import RandAugment, RandomErasing
from tqdm import tqdm
import numpy as np

from model import efficientnet_b0_both
from my_dataset import MyDataSet
from utils import read_split_data_from_folders, train_one_epoch, evaluate, calculate_metrics


# ================== 改进的损失函数：标签平滑 ==================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_probs = nn.functional.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


# ================== 指数移动平均（EMA）提升泛化 ==================
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ================== 改进的训练函数 ==================
def train_one_epoch_improved(model, optimizer, data_loader, device, criterion, scaler=None, ema=None, clip_norm=1.0):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    loop = tqdm(data_loader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # 混合精度训练
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        if ema is not None:
            ema.update()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        loop.set_postfix(loss=loss.item(), acc=total_correct/total_samples)

    return total_loss / len(data_loader), total_correct / total_samples


@torch.no_grad()
def evaluate_improved(model, data_loader, device, criterion, ema=None):
    if ema is not None:
        ema.apply_shadow()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds, all_labels = [], []

    loop = tqdm(data_loader, desc="Validating", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if ema is not None:
        ema.restore()
    avg_loss = total_loss / len(data_loader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc, np.array(all_preds), np.array(all_labels)


def main():
    # ========== 超参数配置 ==========
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据路径（请根据实际修改）
    train_root = "D:/BaiduNetdiskDownload/NEU-CLS-64/NEU-CLS-64/train"
    val_root   = "D:/BaiduNetdiskDownload/NEU-CLS-64/NEU-CLS-64/val"
    save_dir = "./weights_improved"
    os.makedirs(save_dir, exist_ok=True)

    # 训练参数
    batch_size = 32
    epochs = 80                      # 增加训练轮数
    lr = 5e-5                        # 初始学习率降低
    weight_decay = 5e-5              # 适当降低权重衰减
    num_workers = 4                  # 增加数据加载线程
    label_smoothing = 0.1            # 标签平滑系数
    use_amp = True                   # 混合精度训练
    use_ema = True                   # 指数移动平均
    ema_decay = 0.9995
    grad_clip = 1.0
    warmup_epochs = 10               # warmup 轮数

    # ========== 数据增强（大幅增强） ==========
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        RandAugment(num_ops=2, magnitude=9),   # 自动增强
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # 随机擦除
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ========== 数据集准备 ==========
    print("读取数据...")
    train_images_path, train_images_label, val_images_path, val_images_label = \
        read_split_data_from_folders(train_root, val_root)

    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    num_classes = len(class_indices)
    class_names = [None] * num_classes
    for idx_str, name in class_indices.items():
        class_names[int(idx_str)] = name
    print(f"类别数: {num_classes}")
    print(f"类别名称: {class_names}")

    train_dataset = MyDataSet(train_images_path, train_images_label, transform=train_transform)
    val_dataset = MyDataSet(val_images_path, val_images_label, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=train_dataset.collate_fn,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=val_dataset.collate_fn,
                            pin_memory=True)

    # ========== 模型构建（微调注意力参数） ==========
    print("构建双注意力模型（预训练权重）...")
    model = efficientnet_b0_both(
        num_classes=num_classes,
        agent_heads=8,          # 增加 agent 头数
        agent_num=24,           # 增加 agent 数量
        diff_heads=8,           # 增加 differential 头数
        lambda_init=0.5,        # 调整 lambda 初始化
        pretrained=True
    ).to(device)

    # 添加一个额外的 Dropout 层（在注意力后）
    class ImprovedBothAttention(nn.Module):
        def __init__(self, original_model, dropout=0.1):
            super().__init__()
            self.original = original_model
            self.extra_drop = nn.Dropout(dropout)
        def forward(self, x):
            x = self.original(x)
            return self.extra_drop(x)

    # 可选：将模型包装（如果不想要额外 dropout 可以不使用）
    # model = ImprovedBothAttention(model, dropout=0.1)

    # ========== 损失函数（标签平滑） ==========
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

    # ========== 优化器与学习率调度 ==========
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    # 使用余弦退火重启调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    # EMA
    ema = EMA(model, decay=ema_decay) if use_ema else None

    # ========== 训练循环 ==========
    log_data = []
    best_acc = 0.0
    best_val_preds, best_val_labels = None, None

    for epoch in range(1, epochs + 1):
        # Warmup 阶段（前 warmup_epochs 个 epoch 线性增加学习率）
        if epoch <= warmup_epochs:
            warmup_lr = lr * epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            # 非 warmup 时期使用调度器
            if epoch > warmup_epochs:
                scheduler.step()

        # 训练
        train_loss, train_acc = train_one_epoch_improved(
            model, optimizer, train_loader, device, criterion,
            scaler=scaler, ema=ema, clip_norm=grad_clip
        )

        # 验证（使用 EMA 模型）
        val_loss, val_acc, val_preds, val_labels = evaluate_improved(
            model, val_loader, device, criterion, ema=ema
        )

        print(f"[Epoch {epoch:3d}] train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f}  "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")

        log_data.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            torch.save(best_state, os.path.join(save_dir, "best_model_improved.pth"))
            print(f"  -> 保存最佳模型，准确率 {best_acc:.4f}")
            best_val_preds = val_preds
            best_val_labels = val_labels

    # 训练结束，计算详细指标
    print("\n=== 计算最佳模型的详细指标 ===")
    metrics, cm = calculate_metrics(best_val_labels, best_val_preds, class_names)

    # 保存结果到文本文件
    output_file = "training_results_improved.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("===== 改进训练记录 (双注意力 + RandAugment + Label Smoothing + EMA) =====\n")
        f.write("Epoch\tTrain_Loss\tTrain_Acc\tVal_Loss\tVal_Acc\n")
        for entry in log_data:
            f.write(f"{entry['epoch']}\t{entry['train_loss']:.6f}\t{entry['train_acc']:.6f}\t"
                    f"{entry['val_loss']:.6f}\t{entry['val_acc']:.6f}\n")

        f.write("\n========== 混淆矩阵 ==========\n")
        f.write("Rows: True Labels, Columns: Predicted Labels\n\n")
        header = "\t" + "\t".join(class_names)
        f.write(header + "\n")
        for i, row in enumerate(cm):
            row_str = class_names[i] + "\t" + "\t".join(str(x) for x in row)
            f.write(row_str + "\n")

        f.write("\n========== 各类别指标 ==========\n")
        f.write("Class\tPrecision\tRecall(Sensitivity)\tSpecificity\tF1-score\n")
        for cls_name, m in metrics.items():
            f.write(f"{cls_name}\t{m['precision']:.4f}\t{m['recall']:.4f}\t"
                    f"{m['specificity']:.4f}\t{m['f1']:.4f}\n")

        f.write("\n========== 总体指标 ==========\n")
        f.write(f"Best Validation Accuracy: {best_acc:.4f}\n")

    print(f"\n所有结果已保存到 {output_file}")
    print(f"最佳验证准确率: {best_acc:.4f} (目标 ≥ 99%)")


if __name__ == '__main__':
    main()