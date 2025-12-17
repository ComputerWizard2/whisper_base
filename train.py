"""
Fine-tune Whisper model using LoRA for Chinese speech recognition
"""

import os
import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import config


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for speech-to-text models.
    Pads input features and labels.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have different lengths
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def prepare_dataset(batch, processor):
    """
    Prepare a single batch of data for training.
    """
    # Load and resample audio
    audio = batch["audio"]
    
    # Compute log-Mel input features
    batch["input_features"] = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    
    return batch


def compute_metrics(pred, processor, metric):
    """
    Compute WER (Word Error Rate) metric.
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def main():
    """
    Main training function.
    """
    print("=" * 50)
    print("Whisper LoRA Fine-tuning for Chinese ASR")
    print("=" * 50)
    
    # Load processor and model
    print(f"\nLoading model and processor: {config.MODEL_NAME}")
    processor = WhisperProcessor.from_pretrained(
        config.MODEL_NAME, 
        language=config.LANGUAGE, 
        task=config.TASK
    )
    
    model = WhisperForConditionalGeneration.from_pretrained(config.MODEL_NAME)
    
    # Prepare model for training
    model.config.use_cache = False
    # Note: gradient_checkpointing is disabled here to avoid potential issues with LoRA
    # If you encounter memory issues, you can enable it, but it may slow down training
    model.model.encoder.gradient_checkpointing = False
    
    # Configure LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print(f"\nLoading dataset: {config.DATASET_NAME}")
    print(f"Language: {config.DATASET_LANGUAGE}")
    
    try:
        common_voice = DatasetDict()
        common_voice["train"] = load_dataset(
            config.DATASET_NAME,
            config.DATASET_LANGUAGE,
            split="train+validation",
            trust_remote_code=True
        )
        common_voice["test"] = load_dataset(
            config.DATASET_NAME,
            config.DATASET_LANGUAGE,
            split="test",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nNote: You may need to accept the dataset terms at:")
        print(f"https://huggingface.co/datasets/{config.DATASET_NAME}")
        raise
    
    # Limit samples if specified (useful for testing)
    if config.MAX_TRAIN_SAMPLES is not None:
        common_voice["train"] = common_voice["train"].select(range(config.MAX_TRAIN_SAMPLES))
    if config.MAX_EVAL_SAMPLES is not None:
        common_voice["test"] = common_voice["test"].select(range(config.MAX_EVAL_SAMPLES))
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(common_voice['train'])} samples")
    print(f"  Test: {len(common_voice['test'])} samples")
    
    # Cast audio column to correct sampling rate
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=config.SAMPLING_RATE))
    
    # Prepare datasets
    print("\nPreparing datasets...")
    common_voice = common_voice.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=common_voice.column_names["train"],
        num_proc=1  # Use single process to avoid potential issues
    )
    
    # Initialize data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # Load metric
    metric = evaluate.load("wer")
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        evaluation_strategy="steps",
        fp16=config.FP16,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        generation_max_length=config.GENERATION_MAX_LENGTH,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        eval_steps=config.EVAL_STEPS,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        predict_with_generate=config.PREDICT_WITH_GENERATE,
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        logging_dir=config.LOGGING_DIR,
    )
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, metric),
        tokenizer=processor.feature_extractor,
    )
    
    # Start training
    print("\nStarting training...")
    print("=" * 50)
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(config.OUTPUT_DIR)
    processor.save_pretrained(config.OUTPUT_DIR)
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Model saved to: {config.OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
