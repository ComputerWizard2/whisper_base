# 快速开始指南

本指南将帮助您快速开始使用 Whisper LoRA 微调项目。

## 前置要求

- Python 3.8 或更高版本
- CUDA 兼容的 GPU（推荐，但不是必需的）
- 至少 16GB RAM
- 至少 10GB 可用磁盘空间

## 快速安装

### 1. 克隆项目

```bash
git clone https://github.com/ComputerWizard2/whisper_base.git
cd whisper_base
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 安装 ffmpeg

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**MacOS:**
```bash
brew install ffmpeg
```

**Windows:**
从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载并安装

### 5. 登录 Hugging Face

```bash
huggingface-cli login
```

输入您的 Hugging Face token（可以在 https://huggingface.co/settings/tokens 获取）

### 6. 接受数据集条款

访问 [Common Voice 数据集页面](https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1) 并接受使用条款。

## 快速开始训练

### 简单模式（用于测试）

如果您想快速测试代码是否正常工作，可以使用少量样本进行训练：

1. 编辑 `config.py`，设置：
   ```python
   MAX_TRAIN_SAMPLES = 100  # 只使用 100 个训练样本
   MAX_EVAL_SAMPLES = 50    # 只使用 50 个评估样本
   NUM_TRAIN_EPOCHS = 1     # 只训练 1 个 epoch
   ```

2. 运行训练：
   ```bash
   python train.py
   ```

### 完整训练模式

对于完整的模型训练：

1. 确保 `config.py` 中的设置合适（查看注释了解每个参数）

2. 运行训练：
   ```bash
   python train.py
   ```

训练过程可能需要几个小时到一天，具体取决于：
- 您的硬件配置
- 数据集大小
- 训练参数设置

## 监控训练过程

在训练过程中或训练后，您可以使用 TensorBoard 监控：

```bash
tensorboard --logdir ./logs
```

然后在浏览器中打开 http://localhost:6006

## 使用训练好的模型

训练完成后，使用模型转录音频：

```bash
python inference.py --audio your_audio_file.wav
```

## 示例工作流

运行示例工作流脚本，它会引导您完成整个过程：

```bash
python example_workflow.py
```

## 常见问题

### GPU 显存不足

如果遇到 CUDA out of memory 错误：

1. 减小批次大小：
   ```python
   PER_DEVICE_TRAIN_BATCH_SIZE = 4  # 从 8 减小到 4
   PER_DEVICE_EVAL_BATCH_SIZE = 4
   ```

2. 增加梯度累积步数：
   ```python
   GRADIENT_ACCUMULATION_STEPS = 4  # 从 2 增加到 4
   ```

### 数据集下载失败

确保：
1. 网络连接正常
2. 已登录 Hugging Face
3. 已接受数据集使用条款

### 训练速度慢

优化建议：
1. 使用 GPU（自动检测）
2. 启用混合精度训练（默认已启用）
3. 增加 `DATALOADER_NUM_WORKERS`（根据 CPU 核心数）
4. 减少数据集大小（用于快速迭代）

## 测试数据

如果您需要测试音频文件，可以：

1. 从 [Common Voice](https://commonvoice.mozilla.org/zh-CN/datasets) 下载样本
2. 使用您自己的中文音频文件
3. 使用在线文本转语音服务生成测试音频

## 下一步

- 阅读 [README.md](README.md) 了解更多详细信息
- 调整 `config.py` 中的参数以优化性能
- 尝试不同的 LoRA 配置
- 在您自己的数据集上训练

## 获取帮助

如果遇到问题：
1. 查看 [README.md](README.md) 中的"常见问题"部分
2. 在 GitHub 上提交 Issue
3. 查看相关文档链接

祝您使用愉快！🚀
