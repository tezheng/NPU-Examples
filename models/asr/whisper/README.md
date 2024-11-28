# Running Whisper-Base-En on QNPU Devices

## 1. Install FFmpeg

Ensure FFmpeg is installed to handle audio conversion tasks. You can use tools like `winget`, `scoop`, or any preferred method for installation.

```powershell
# Using winget
winget install -a arm64 ffmpeg

# Using scoop
scoop install --arch arm64 ffmpeg
```

## 2. Set Up Conda Environment

Create the necessary Conda environment using your preferred Conda toolchain (`miniconda`, `mamba`, `micromamba`, etc.).

```powershell
conda env create -f environment.yml
conda activate whisper-qnpu-py310
```

## 3. Quantize Whisper-Base-En with Qualcomm AI Hub

Quantize the `whisper_base_en` model for QNPU. Using `--skip-inferencing` and `--skip-profiling` skips performance measurement, remove these flags if you want to benchmark the model during quantization.

```powershell
# Set up your Qualcomm AI Hub development environment
qai-hub configure --api_token ${QAI_API_TOKEN}

# Convert and quantize the model
python export.py

# or with
python -m qai_hub_models.models.whisper_base_en.export \
    --skip-inferencing \
    --skip-profiling \
    --device "Snapdragon X Elite CRD" \
    --target-runtime onnx \
    --output-dir build
```

## 4. Run the Demo

Ensure your audio file is in mono format before running the demo. You can convert stereo audio to mono using FFmpeg:

```powershell
# Convert stereo to mono with 22050 Hz sample rate
ffmpeg -i input.mp3 -ar 22050 -ac 1 output.mp3
```

Run the demo script with your prepared audio file:

```powershell
python demo.py --audio_file /path/to/audio_sample.mp3
```

---

### Notes

- **FFmpeg Installation**: Refer to the FFmpeg documentation for alternative installation methods if needed.
- **Audio Requirements**: The demo requires mono audio files with a sample rate of 22050 Hz.
- **Quantization**: Using `--skip-inferencing` and `--skip-profiling` skips performance measurement. Remove these flags if you want to benchmark the model during quantization.

This README is now concise, consistent, and easy to follow. Let me know if further customization is needed!
