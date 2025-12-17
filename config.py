"""
Configuration file for Whisper LoRA fine-tuning
"""

# Model configuration
MODEL_NAME = "openai/whisper-base"
LANGUAGE = "zh"  # Chinese
TASK = "transcribe"

# LoRA configuration
LORA_R = 8  # Rank of the LoRA matrices
LORA_ALPHA = 16  # Scaling factor
LORA_DROPOUT = 0.1  # Dropout probability
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # Modules to apply LoRA to

# Dataset configuration
DATASET_NAME = "mozilla-foundation/common_voice_16_1"
DATASET_LANGUAGE = "zh-CN"
MAX_AUDIO_LENGTH = 30.0  # seconds
SAMPLING_RATE = 16000

# Training configuration
OUTPUT_DIR = "./whisper-base-lora-chinese"
LOGGING_DIR = "./logs"
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-3
WARMUP_STEPS = 500
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 100
FP16 = True  # Use mixed precision training
DATALOADER_NUM_WORKERS = 4

# Evaluation configuration
PREDICT_WITH_GENERATE = True
GENERATION_MAX_LENGTH = 225

# Data preprocessing
MAX_TRAIN_SAMPLES = None  # Set to a number to limit training samples for testing
MAX_EVAL_SAMPLES = None  # Set to a number to limit evaluation samples for testing
