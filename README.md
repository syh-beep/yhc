# EfficientNet-DualAttention-for-Hot-Rolled-Steel-Defect-Detection



> 官方PyTorch实现 | 论文处于投刊阶段，标题：《Multi-class Surface Defect Recognition for Hot-rolled Steel Strips Based on EfficientNet and Dual Attention》  


> 提出EfficientNet-DualAttention模型，基于PyTorch框架实现六类热轧钢带表面缺陷的高精度识别，助力工业质检智能化诊断。  


## 1. 研究背景与模型定位  

热轧钢带表面缺陷（裂纹、夹杂、斑块、点蚀、氧化铁皮压入、划痕）直接影响钢材质量与服役安全。传统人工检测效率低、依赖经验；现有深度学习方法在高精度与实时性之间难以平衡，且对低对比度、类间相似性高的缺陷识别能力不足。

为此，本项目以轻量级 EfficientNet-B0 为主干，在最后一个 1×11×1 卷积后、GAP 前依次引入两种注意力机制：
Agent Attention：通过少量可学习代理令牌（agent tokens）全局聚合上下文语义，增强模型对氧化铁皮、油污等复杂背景的抑制能力。
Differential Attention：通过双分支差分运算显式建模特征通道间的二阶响应差异，强化微裂纹、模糊边缘及低对比度缺陷的细节响应。
二者级联协同，实现“全局语义引导 + 局部细节锐化”的联合优化。
  


## 2.  EfficientNet-DualAttention 核心创新点  

 
1.双重注意力协同机制
Agent Attention 负责全局上下文建模与背景抑制；
Differential Attention 负责局部细节锐化与边缘增强；
级联集成后显著提升复杂工况下的缺陷分类精度。


2. 轻量化主干与高效注意力
保留 EfficientNet-B0 的复合缩放策略，参数量约 13.2M，计算量约 2.6 GFLOPs；
Agent Attention 将标准自注意力的 O(N2)O(N2) 复杂度降低至近似线性；
Differential Attention 以轻量差分运算增强细粒度特征，不显著增加计算开销。



3. 训练策略优化
标签平滑（Label Smoothing, 0.1）
指数移动平均（EMA, decay=0.9995）
混合精度训练（AMP）
梯度裁剪（clip_norm=1.0）
Warmup + CosineAnnealingWarmRestarts 学习率调度
RandAugment、RandomErasing、颜色抖动等强数据增强





## 3. 实验数据集：NEU-CLS 热轧钢表面缺陷数据集


### 3.1 数据集概况  

本研究基于NEU-CLS 热轧钢表面缺陷数据集，包含六种常见热轧钢表面缺陷类型，数据集为公开数据集，可从官方渠道获取使用：  

| 数据集名称 | 包含类别 | 图像总数 | 图像分辨率 | 数据分布（训练:验证:测试） |
|------------|-------------------------|----------|------------|-----------------------|
| NEU-CLS 热轧钢表面缺陷数据集 | 裂纹（Crazing）、夹杂（Inclusion）、斑块（Patches）、点蚀（Pitted Surface）、氧化铁皮压入（Rolled-in Scale）、划痕（Scratches）| 6,000+ | 统一resize至224×224（适配模型输入） | 3:1:1 |  



2. **文件夹组织**（下载后解压至项目根目录，结构如下）：  
```  
NEU-CLS-64/├── train/                  # 训练集
│   ├── cr/                 # 裂纹（Crazing）图像
│   ├── in/                 # 夹杂（Inclusion）图像
│   ├── pa/                 # 斑块（Patches）图像
│   ├── ps/                 # 点蚀（Pitted Surface）图像
│   ├── rs/                 # 氧化铁皮压入（Rolled-in Scale）图像
│   └── sc/                 # 划痕（Scratches）图像
├── val/                    # 验证集
│   ├── cr/
│   ├── in/
│   ├── pa/
│   ├── ps/
│   ├── rs/
│   └── sc/
└── pre/                    # 待预测图像目录（可选）    
├── cr/    
├── in/    
├── pa/    
├── ps/    
├── rs/    
└── sc/
            
```  


## 4. 实验环境配置  

### 4.1 依赖安装  

推荐使用Anaconda创建虚拟环境，确保依赖版本匹配（PyTorch框架核心依赖）：  

```bash  
# 1. 创建并激活虚拟环境  
conda create -n EfficientNet-DualAttention-torch python=3.9
conda activate EfficientNet-DualAttention-torch 
  

# 2. 安装PyTorch（支持GPU/CPU，示例为CPU版本）  
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu

# 3. 安装其他依赖库   
pip install numpy==1.24.4 matplotlib==3.5.2 opencv-python==4.11.0.86
pip install pandas==1.4.4 pillow==11.1.0 scikit-learn==1.0.2
pip install tqdm==4.67.1 tensorboard==2.11.2

```  

## 5 代码使用说明  

### 5.1 模型训练  

运行`train.py`脚本启动训练，支持通过参数调整训练配置，示例命令：  

```bash  
# 训练双注意力模型（EfficientNet-B0 + Agent Attention + Differential Attention）
python train.py \
  --pretrained \ 
--agent_heads 8 \  
--agent_num 24 \  
--diff_heads 8 \  
--lambda_init 0.5 \
  --data_path ./NEU-CLS-64 \ # 数据集根目录
  --batch_size 32 \  
--epochs 80 \  
--lr 5e-5 \  
--weight_decay 5e-5 \  
--label_smoothing 0.1 \  
--use_amp \  
--use_ema \  
--warmup_epochs 10 \  
--device cuda
#消融实验
python bestboth.py \
  --weight_path ./weights/best_model.pth \  # 双注意力权重路径
--data_path ./NEU-CLS-64 \  # 数据集根目录
  --agent_heads 8 \  
--agent_num 24 \  
--device cuda
```  


#### 关键参数说明：  

| 参数名 | 含义 | 默认值 |
|-----------------|---------------------------------------|-----------------|
| `--data_dir` | 数据集根目录路径 | `./datasets` |
| `--epochs` | 训练轮数 | 80 |
| `--batch_size` | 批次大小（根据显存调整，8/16/32） |32 |
| `--lr` | 初始学习率 | 5e-5 |
| `--save_dir` | 训练权重保存目录（.pth格式） | ./weights_improved |
| `--device` | 训练设备（`GPU`或`CPU`） | `CPU` |  



### 5.2 模型预测  

使用训练好的权重进行热轧钢表面缺陷图像预测，运行`predict.py`脚本，示例命令：  

```bash  
python predict.py\
--image-path ./NEU-CLS-64/pre \  # 输入图像路径
--weight_path ./weights/best_model.pth \   # 预训练权重路径
--device CPU  
```  


#### 预测输出示例：  

```  
输入图像路径： ./NEU-CLS-64/pre/Crazing/Crazing_1.jpg\  
预测类别：裂纹（Crazing）
置信度：0.9883 
  
```  


## 6. 项目文件结构  

```  
EfficientNet-DualAttention/
├── datasets/  # 六类热轧钢表面缺陷数据集（NEU-CLS）    
├── model.py         ## 整体模型实现（EfficientNet-B0 + Agent Attention + Differential Attention）
├──efficientnet.py             #原 EfficientNet-B0 主干实现
├── agent_attention.py          #Agent Attention 全局上下文建模实现  
├── differential_attention.py              # Differential Attention 细节特征锐化实现
  
├── my_dataset.py        #自定义数据加载器 
├── class_indices.json      #数据集的类别及类别数
├── utils.py               #训练工具函数集   
├── train.py              # 模型训练脚本（PyTorch版）  
├── predict.py              # 模型预测脚本（PyTorch版）
├── bestboth.py           # 消融实验脚本 
├── best_model.pth              # 模型训练权重   
└── README.md             # 项目说明文档（本文档）  
```  


## 7. 已知问题与注意事项  

1. **框架适配**：本项目基于 PyTorch 2.0.1+ 和 EfficientNet-B0 卷积神经网络架构；
2.**输入尺寸**：模型固定输入为 224×224×3，训练和预测时会自动进行 resize 和中心裁剪；
3.**模块配置**：支持通过命令行参数灵活启用 Agent Attention 全局上下文建模和 Differential Attention 细节特征锐化；
4.**数据集扩展**：如需新增类别，需补充图像数据并修改数据集目录结构或 class_indices.json 文件，模型会自动读取并更新类别数；
5.**训练策略**：支持迁移学习、标签平滑、指数移动平均、混合精度训练以及 Warmup 与余弦退火学习率调度。

## 8. 引用与联系方式  

### 8.1 引用方式  

论文处于投刊阶段，正式发表后将更新BibTeX引用格式，当前可临时引用：  

```bibtex  
@article{EfficientNet_DualAttention_SteelDefect,
  
title={Multi-class Surface Defect Recognition for Hot-rolled Steel Strips Based on EfficientNet and Dual Attention},  
  author={[作者姓名，待发表时补充]},  
  journal={[期刊名称，待录用后补充]},  
  year={2026},  
  note={Manuscript submitted for publication}  
}  
```  


### 8.2 联系方式  

若遇到代码运行问题或学术交流需求，请联系：  
- 邮箱：huangshiyu@huuc.edu.cn 
- GitHub Issue：直接在本仓库提交Issue，会在1-3个工作日内回复。
