"""
Data preparation and exploration script for Common Voice dataset
"""

import argparse
from datasets import load_dataset
import config


def explore_dataset(dataset_name, language, split="train", num_samples=5):
    """
    Explore the dataset and display sample information.
    
    Args:
        dataset_name: Name of the dataset
        language: Language code
        split: Dataset split (train/validation/test)
        num_samples: Number of samples to display
    """
    print("=" * 50)
    print(f"Exploring {dataset_name} - {language}")
    print("=" * 50)
    
    try:
        # Load dataset
        print(f"\nLoading {split} split...")
        dataset = load_dataset(
            dataset_name,
            language,
            split=split,
            trust_remote_code=True
        )
        
        print(f"\nDataset size: {len(dataset)} samples")
        print(f"\nDataset features: {dataset.features}")
        
        # Display sample data
        print(f"\n{'-' * 50}")
        print(f"Displaying {num_samples} sample(s):")
        print("-" * 50)
        
        for i, sample in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
            print(f"\nSample {i + 1}:")
            print(f"  Sentence: {sample.get('sentence', 'N/A')}")
            if 'audio' in sample:
                audio_info = sample['audio']
                print(f"  Audio path: {audio_info.get('path', 'N/A')}")
                if 'array' in audio_info:
                    print(f"  Audio array shape: {len(audio_info['array'])}")
                print(f"  Sampling rate: {audio_info.get('sampling_rate', 'N/A')} Hz")
            print("-" * 50)
        
        return dataset
        
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nNote: You may need to:")
        print("1. Accept the dataset terms at:")
        print(f"   https://huggingface.co/datasets/{dataset_name}")
        print("2. Login to Hugging Face: huggingface-cli login")
        return None


def check_dataset_splits(dataset_name, language):
    """
    Check available splits in the dataset.
    
    Args:
        dataset_name: Name of the dataset
        language: Language code
    """
    print("\nChecking available splits...")
    
    splits = ["train", "validation", "test"]
    available_splits = []
    
    for split in splits:
        try:
            ds = load_dataset(
                dataset_name,
                language,
                split=split,
                trust_remote_code=True,
                streaming=True
            )
            # Try to get first sample to verify split exists
            next(iter(ds))
            available_splits.append(split)
            print(f"  ✓ {split}")
        except Exception:
            print(f"  ✗ {split} (not available)")
    
    return available_splits


def main():
    """
    Main function for data preparation.
    """
    parser = argparse.ArgumentParser(description="Explore Common Voice dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default=config.DATASET_NAME,
        help=f"Dataset name (default: {config.DATASET_NAME})"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=config.DATASET_LANGUAGE,
        help=f"Language code (default: {config.DATASET_LANGUAGE})"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to explore (default: train)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to display (default: 5)"
    )
    parser.add_argument(
        "--check_splits",
        action="store_true",
        help="Check available splits in the dataset"
    )
    
    args = parser.parse_args()
    
    if args.check_splits:
        check_dataset_splits(args.dataset, args.language)
    else:
        explore_dataset(args.dataset, args.language, args.split, args.num_samples)


if __name__ == "__main__":
    main()
