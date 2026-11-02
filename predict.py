import os
import json
import csv
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 导入双注意力模型
from model import efficientnet_b0_both as create_model


def get_image_files(root_dir, extensions=('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
    image_paths = []
    labels = []
    for subdir, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(extensions):
                img_path = os.path.join(subdir, f)
                image_paths.append(img_path)
                parent_dir = os.path.basename(subdir)
                labels.append(parent_dir)
    return image_paths, labels


def main():
    pre_folder = "D:/BaiduNetdiskDownload/NEU-CLS-64/NEU-CLS-64/pre"
    weights_path = "./weights/best_model.pth"
    class_json_path = "./class_indices.json"
    output_csv = "./pre_predictions.csv"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    assert os.path.exists(class_json_path), f"Class json not found: {class_json_path}"
    with open(class_json_path, 'r') as f:
        class_indices = json.load(f)
    idx_to_name = {int(k): v for k, v in class_indices.items()}
    num_classes = len(class_indices)
    print(f"Loaded {num_classes} classes: {list(idx_to_name.values())}")

    name_to_idx = {v: k for k, v in idx_to_name.items()}

    # 创建模型（与训练时一致，pretrained=False 因为只加载权重）
    model = create_model(num_classes=num_classes, pretrained=False).to(device)
    assert os.path.exists(weights_path), f"Model weights not found: {weights_path}"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # 预处理（与验证集一致）
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_paths, folder_names = get_image_files(pre_folder)
    if not image_paths:
        print(f"No images found in {pre_folder}")
        return

    print(f"Found {len(image_paths)} images.")

    predictions = []
    correct_count = 0
    total_with_label = 0

    for img_path, folder_name in tqdm(zip(image_paths, folder_names), total=len(image_paths), desc="Predicting"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = data_transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.softmax(output, dim=1)
                pred_idx = torch.argmax(prob, dim=1).item()
                pred_name = idx_to_name[pred_idx]
                confidence = prob[0, pred_idx].item()

            predictions.append({
                "image_path": img_path,
                "pred_class": pred_name,
                "confidence": confidence,
                "folder_name": folder_name
            })

            if folder_name in name_to_idx:
                total_with_label += 1
                if pred_idx == name_to_idx[folder_name]:
                    correct_count += 1

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "folder_name", "pred_class", "confidence"])
        writer.writeheader()
        for p in predictions:
            writer.writerow({
                "image_path": p["image_path"],
                "folder_name": p["folder_name"],
                "pred_class": p["pred_class"],
                "confidence": f"{p['confidence']:.4f}"
            })
    print(f"\nPredictions saved to {output_csv}")

    if total_with_label > 0:
        overall_acc = correct_count / total_with_label
        print(f"\n=== 准确率统计 ===")
        print(f"Total images with label: {total_with_label}")
        print(f"Correct predictions: {correct_count}")
        print(f"Overall accuracy: {overall_acc:.2%}")


if __name__ == '__main__':
    main()