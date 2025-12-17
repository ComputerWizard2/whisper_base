"""
Inference script for transcribing audio files using fine-tuned Whisper model
"""

import os
import sys
import torch
import argparse
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import librosa
import numpy as np

import config


def load_audio(audio_path, target_sr=16000):
    """
    Load and preprocess audio file.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sampling rate (default: 16000)
    
    Returns:
        audio array and sampling rate
    """
    print(f"Loading audio from: {audio_path}")
    
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    print(f"Audio duration: {len(audio) / sr:.2f} seconds")
    print(f"Sampling rate: {sr} Hz")
    
    return audio, sr


def transcribe_audio(model, processor, audio, sampling_rate):
    """
    Transcribe audio using the model.
    
    Args:
        model: Fine-tuned Whisper model
        processor: Whisper processor
        audio: Audio array
        sampling_rate: Sampling rate of audio
    
    Returns:
        Transcribed text
    """
    # Prepare input features
    input_features = processor(
        audio, 
        sampling_rate=sampling_rate, 
        return_tensors="pt"
    ).input_features
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_features = input_features.to(device)
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    
    # Decode transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription


def main():
    """
    Main inference function.
    """
    parser = argparse.ArgumentParser(description="Transcribe audio using fine-tuned Whisper model")
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file to transcribe"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=config.OUTPUT_DIR,
        help=f"Path to fine-tuned model (default: {config.OUTPUT_DIR})"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=config.MODEL_NAME,
        help=f"Base model name (default: {config.MODEL_NAME})"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Whisper Audio Transcription")
    print("=" * 50)
    
    # Check if audio file exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        print(f"\nPlease train the model first using: python train.py")
        sys.exit(1)
    
    # Load processor
    print(f"\nLoading processor from: {args.model_path}")
    processor = WhisperProcessor.from_pretrained(
        args.model_path,
        language=config.LANGUAGE,
        task=config.TASK
    )
    
    # Load base model
    print(f"Loading base model: {args.base_model}")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    
    # Load LoRA weights
    print(f"Loading LoRA weights from: {args.model_path}")
    model = PeftModel.from_pretrained(model, args.model_path)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    # Load audio
    print("\n" + "-" * 50)
    audio, sr = load_audio(args.audio, target_sr=config.SAMPLING_RATE)
    
    # Transcribe
    print("\nTranscribing...")
    transcription = transcribe_audio(model, processor, audio, sr)
    
    # Display result
    print("\n" + "=" * 50)
    print("Transcription Result:")
    print("=" * 50)
    print(transcription)
    print("=" * 50)


if __name__ == "__main__":
    main()
