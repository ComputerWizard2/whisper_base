# 基于 LoRA 微调 Whisper 模型实现中文语音识别

## 项目简介

本项目旨在利用 LoRA (Low-Rank Adaptation) 技术对 OpenAI 的 Whisper base 模型进行微调，以实现高效的中文语音识别 (ASR)。通过在 Common Voice 中文数据集上进行训练，并使用 LoRA 显著减少了可训练参数数量，从而加速了训练过程并降低了计算资源消耗。最终模型将用于对测试音频文件进行转录。

## 主要特性

- ✨ 使用 LoRA 技术减少可训练参数，降低训练成本
- 🚀 基于 Whisper base 模型进行中文语音识别
- 📊 在 Common Voice 中文数据集上训练
- 🎯 支持音频文件转录和推理
- ⚡ 支持 GPU 加速训练和推理
- 📈 集成 TensorBoard 用于训练监控

## 环境配置

### 依赖库

项目依赖以下库：

```
kagglehub==0.3.13
datasets==3.2.0
librosa==0.10.2.post1
peft==0.14.0
torch==2.6.0
torchaudio==2.6.0
torchvision==0.21.0
ffmpeg-python==0.2.0
transformers==4.51.3
```

### 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/ComputerWizard2/whisper_base.git
cd whisper_base
```

2. 创建虚拟环境（推荐）：

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

4. 安装 ffmpeg（用于音频处理）：

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

5. 验证安装：

```bash
python validate_setup.py
```

此脚本将检查所有依赖是否正确安装。

## 项目结构

```
whisper_base/
├── README.md                 # 项目说明文档
├── QUICKSTART.md             # 快速开始指南
├── requirements.txt          # 依赖库列表
├── config.py                 # 配置文件（模型、训练参数等）
├── train.py                  # 训练脚本
├── inference.py              # 推理脚本
├── prepare_data.py           # 数据准备和探索脚本
├── example_workflow.py       # 示例工作流程脚本
├── validate_setup.py         # 安装验证脚本
└── whisper-base-lora-chinese/  # 训练输出目录（训练后生成）
```

## 使用说明

### 1. 数据准备

在开始训练之前，可以使用 `prepare_data.py` 脚本探索数据集：

```bash
# 查看训练集的前 5 个样本
python prepare_data.py --split train --num_samples 5

# 检查可用的数据集分割
python prepare_data.py --check_splits

# 查看测试集
python prepare_data.py --split test --num_samples 3
```

**注意：** 首次使用需要：
1. 在 [Hugging Face](https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1) 上接受数据集使用条款
2. 登录 Hugging Face：`huggingface-cli login`

### 2. 配置参数

在 `config.py` 中可以修改以下参数：

- **模型配置：** `MODEL_NAME`, `LANGUAGE`, `TASK`
- **LoRA 配置：** `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`
- **训练配置：** `NUM_TRAIN_EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE` 等
- **数据集配置：** `DATASET_NAME`, `DATASET_LANGUAGE`

### 3. 训练模型

运行训练脚本开始微调：

```bash
python train.py
```

训练过程中将：
- 自动下载 Whisper base 模型
- 加载 Common Voice 中文数据集
- 应用 LoRA 配置并冻结基础模型
- 开始训练并定期保存检查点
- 在测试集上评估模型性能（WER）

训练输出：
- 模型权重保存在 `./whisper-base-lora-chinese/` 目录
- TensorBoard 日志保存在 `./logs/` 目录

查看训练日志：
```bash
tensorboard --logdir ./logs
```

### 4. 推理和转录

使用训练好的模型对音频文件进行转录：

```bash
# 基本使用
python inference.py --audio /path/to/your/audio.wav

# 指定模型路径
python inference.py --audio /path/to/your/audio.wav --model_path ./whisper-base-lora-chinese

# 指定基础模型
python inference.py --audio /path/to/your/audio.wav --base_model openai/whisper-base
```

支持的音频格式：
- WAV, MP3, FLAC, OGG 等常见格式
- 自动重采样到 16kHz

## LoRA 技术说明

LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法：

- **原理：** 在预训练模型的特定层中插入低秩矩阵，只训练这些新增的参数
- **优势：**
  - 大幅减少可训练参数数量（通常减少 90% 以上）
  - 降低显存占用和训练时间
  - 保持基础模型性能的同时实现任务适配
  - 便于模型部署和分享

在本项目中，LoRA 应用于 Whisper 模型的注意力层（`q_proj` 和 `v_proj`），通过少量参数即可实现中文语音识别的高效微调。

## 性能指标

模型性能通过 WER (Word Error Rate，词错误率) 评估：

- **WER** 越低表示识别准确率越高
- 训练过程中会在测试集上定期评估 WER
- 最佳模型会自动保存

## 常见问题

### 1. 显存不足

如果遇到显存不足的问题，可以尝试：
- 减小 `PER_DEVICE_TRAIN_BATCH_SIZE` 和 `PER_DEVICE_EVAL_BATCH_SIZE`
- 增加 `GRADIENT_ACCUMULATION_STEPS`
- 在 `config.py` 中设置 `MAX_TRAIN_SAMPLES` 限制训练样本数量

### 2. 数据集访问问题

确保已经：
1. 接受 Common Voice 数据集的使用条款
2. 使用 `huggingface-cli login` 登录

### 3. ffmpeg 相关错误

确保系统已安装 ffmpeg：
```bash
ffmpeg -version
```

### 4. 训练速度慢

建议：
- 使用 GPU 进行训练（自动检测）
- 启用混合精度训练（`FP16 = True`）
- 调整 `DATALOADER_NUM_WORKERS` 增加数据加载速度

## 技术栈

- **PyTorch**: 深度学习框架
- **Transformers**: Hugging Face 提供的预训练模型库
- **PEFT**: 参数高效微调库（LoRA 实现）
- **Datasets**: Hugging Face 数据集库
- **Librosa**: 音频处理库
- **Evaluate**: 模型评估库

## 许可证

本项目遵循 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 参考资料

- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Common Voice Dataset](https://commonvoice.mozilla.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。
