"""
Example script demonstrating the complete workflow
This script shows how to use the project step by step
"""

import os
import sys


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def main():
    """
    Demonstrate the complete workflow.
    """
    print_section("Whisper LoRA Fine-tuning - Complete Workflow")
    
    print("This script demonstrates the complete workflow for fine-tuning")
    print("Whisper model using LoRA for Chinese speech recognition.\n")
    
    # Step 1: Environment setup
    print_section("Step 1: Environment Setup")
    print("Before starting, ensure you have:")
    print("  1. Installed all dependencies: pip install -r requirements.txt")
    print("  2. Installed ffmpeg on your system")
    print("  3. Logged in to Hugging Face: huggingface-cli login")
    print("  4. Accepted Common Voice dataset terms")
    print("\nPress Enter to continue...")
    input()
    
    # Step 2: Data exploration
    print_section("Step 2: Explore Dataset")
    print("First, let's explore the Common Voice Chinese dataset:\n")
    print("Command: python prepare_data.py --split train --num_samples 3\n")
    
    response = input("Do you want to run this command? (y/n): ")
    if response.lower() == 'y':
        os.system("python prepare_data.py --split train --num_samples 3")
    
    # Step 3: Configuration
    print_section("Step 3: Configure Training Parameters")
    print("Review and modify config.py if needed:")
    print("  - Model: openai/whisper-base")
    print("  - Language: Chinese (zh)")
    print("  - LoRA rank (r): 8")
    print("  - Training epochs: 3")
    print("  - Batch size: 8")
    print("\nYou can adjust these parameters based on your GPU memory.")
    print("\nPress Enter to continue...")
    input()
    
    # Step 4: Training
    print_section("Step 4: Train the Model")
    print("Start training the model with LoRA:\n")
    print("Command: python train.py\n")
    print("⚠️  Warning: Training may take several hours depending on:")
    print("  - Dataset size")
    print("  - GPU availability")
    print("  - Batch size and other hyperparameters")
    print("\nThe model will be saved to: ./whisper-base-lora-chinese/")
    print("\nMonitor training with TensorBoard:")
    print("  tensorboard --logdir ./logs\n")
    
    response = input("Do you want to start training? (y/n): ")
    if response.lower() == 'y':
        print("\nStarting training...")
        os.system("python train.py")
    else:
        print("\nSkipping training. You can run it manually later.")
    
    # Step 5: Inference
    print_section("Step 5: Test the Model")
    print("After training, use the model to transcribe audio:\n")
    print("Command: python inference.py --audio /path/to/audio.wav\n")
    print("Example:")
    print("  python inference.py --audio test_audio.wav\n")
    
    audio_path = input("Enter path to an audio file (or press Enter to skip): ")
    if audio_path and os.path.exists(audio_path):
        print(f"\nTranscribing {audio_path}...")
        os.system(f"python inference.py --audio {audio_path}")
    elif audio_path:
        print(f"\n⚠️  File not found: {audio_path}")
    else:
        print("\nSkipping inference demo.")
    
    # Summary
    print_section("Summary")
    print("Workflow complete! Here's what you can do next:\n")
    print("1. Monitor training progress:")
    print("   tensorboard --logdir ./logs\n")
    print("2. Resume training from checkpoint:")
    print("   Modify config.py OUTPUT_DIR to existing checkpoint\n")
    print("3. Transcribe audio files:")
    print("   python inference.py --audio your_audio.wav\n")
    print("4. Adjust hyperparameters in config.py for better results\n")
    print("For more information, check README.md")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
        sys.exit(0)
