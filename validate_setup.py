"""
Setup validation script
Checks if all dependencies are installed correctly
"""

import sys
import subprocess


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    else:
        print("✅ Python version is compatible")
        return True


def check_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False


def check_ffmpeg():
    """Check if ffmpeg is installed."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ ffmpeg is installed: {version_line}")
            return True
        else:
            print("❌ ffmpeg is not working correctly")
            return False
    except FileNotFoundError:
        print("❌ ffmpeg is not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking ffmpeg: {e}")
        return False


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA is available: {device_count} device(s)")
            print(f"   Device 0: {device_name}")
            return True
        else:
            print("⚠️  CUDA is not available (CPU only)")
            return False
    except ImportError:
        print("⚠️  Cannot check CUDA (torch not installed)")
        return False


def main():
    """
    Main validation function.
    """
    print("=" * 60)
    print("  Whisper LoRA Setup Validation")
    print("=" * 60)
    print()
    
    results = []
    
    # Check Python version
    print("Checking Python version...")
    results.append(check_python_version())
    print()
    
    # Check required packages
    print("Checking required packages...")
    packages = [
        ("datasets", "datasets"),
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("librosa", "librosa"),
        ("evaluate", "evaluate"),
        ("kagglehub", "kagglehub"),
    ]
    
    for package_name, import_name in packages:
        results.append(check_package(package_name, import_name))
    print()
    
    # Check ffmpeg
    print("Checking ffmpeg...")
    ffmpeg_ok = check_ffmpeg()
    results.append(ffmpeg_ok)
    print()
    
    # Check CUDA
    print("Checking CUDA availability...")
    cuda_ok = check_cuda()
    print()
    
    # Summary
    print("=" * 60)
    print("  Validation Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} required checks")
    
    if passed == total:
        print("\n✅ All required dependencies are installed!")
        if not cuda_ok:
            print("\n⚠️  Note: CUDA is not available. Training will use CPU (slower).")
        print("\nYou can now start training:")
        print("  python train.py")
    else:
        print("\n❌ Some dependencies are missing.")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        
        if not ffmpeg_ok:
            print("\nTo install ffmpeg:")
            print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
            print("  MacOS: brew install ffmpeg")
            print("  Windows: Download from https://ffmpeg.org/download.html")
    
    print("\n" + "=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
