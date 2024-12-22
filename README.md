# STFT-ISTFT-ONNX
Export the Short-Time Fourier Transform (STFT) or Inverse Short-Time Fourier Transform (ISTFT) processes to ONNX format.

## Introduction
This repository provides tools to export your custom STFT or ISTFT processes in ONNX format. Follow the Python script instructions and configure the parameters as needed.

## Key Features
- Export STFT and ISTFT processes in ONNX format.
- Supports configurable parameters for audio processing, including window type, FFT components, and Mel bands.
- Provides high accuracy compared to PyTorch's native STFT/ISTFT methods.

## Below is an example of how to use this library:
---
```python
from STFT_Process import STFT_Process
from pydub import AudioSegment
import torch
import soundfile as sf


test_audio = './audio_file.mp3'                     # Specify the path to your audio file.
save_reconstructed_audio = './saved_audio.wav'      # Save the reconstructed.


# Configuration Parameters
MAX_SIGNAL_LENGTH = 1024        # Maximum number of frames for audio length after STFT. Use larger values for long audio inputs (e.g., 4096).
INPUT_AUDIO_LENGTH = 5120       # Length of the audio input signal (in samples) for static axis export. Should be a multiple of NFFT.
WINDOW_TYPE = 'hann'            # Type of window function used in the STFT.
N_MELS = 100                    # Number of Mel bands for the Mel-spectrogram.
NFFT = 512                      # Number of FFT components for the STFT process.
HOP_LENGTH = 128                # Number of samples between successive frames in the STFT.
SAMPLE_RATE = 16000             # Target sample rate.
STFT_TYPE = "stft_B"            # stft_A: output real_part only;  stft_B: outputs real_part & imag_part
ISTFT_TYPE = "istft_B"          # istft_A: Inputs = [magnitude, phase];  istft_B: Inputs = [magnitude, real_part, imag_part], The dtype of imag_part is float format.

# Load the audio
audio = torch.tensor(
    AudioSegment.from_file(test_audio)
    .set_channels(1)
    .set_frame_rate(SAMPLE_RATE)
    .get_array_of_samples(), 
    dtype=torch.float32
)
audio = audio.reshape(1, 1, -1)
audio_parts = audio[:, :, :INPUT_AUDIO_LENGTH]

# Create the STFT model
custom_stft = STFT_Process(
    model_type=STFT_TYPE, 
    n_fft=NFFT, 
    n_mels=N_MELS, 
    hop_len=HOP_LENGTH, 
    max_frames=0,  # Not important here.
    window_type=WINDOW_TYPE
).eval()

# Process the audio (STFT)
real_part, imag_part = custom_stft(audio_parts, 'constant')  # pad_mode options: ['constant', 'reflect']

# Calculate the magnitude
magnitude = torch.sqrt(real_part**2 + imag_part**2)

# Create the ISTFT model
custom_istft = STFT_Process(
    model_type=ISTFT_TYPE, 
    n_fft=NFFT, 
    n_mels=N_MELS, 
    hop_len=HOP_LENGTH, 
    max_frames=MAX_SIGNAL_LENGTH, 
    window_type=WINDOW_TYPE
).eval()

# Reconstruct the audio from magnitude and phase
audio_reconstructed = custom_istft(magnitude, real_part, imag_part).to(torch.int16)
sf.write(save_reconstructed_audio, audio_reconstructed[0, 0], SAMPLE_RATE, format='WAVEX')
```

---

## 比较差异 Comparison to `torch.stft()` and `torch.istft()`

### Performance Metrics
| OS            | Device       | Backend    | Model       | Window Type | Absolute Mean Difference | Real-Time Factor<br>Chunk Size: 160,000 or 10s |
|:-------------:|:------------:|:----------:|:-----------:|:-----------:|:-------------------------:|:--------------------------------------------:|
| Ubuntu 24.04  | Laptop       | CPU i5-7300HQ | `stft_A` f32 | `hann`      | 3.8567e-05               | 0.0005                                       |
| Ubuntu 24.04  | Laptop       | CPU i5-7300HQ | `istft_A` f32 | `hann`      | 1.3480e-05               | 0.0015                                       |
| Ubuntu 24.04  | Laptop       | CPU i5-7300HQ | `stft_B` f32 | `hann`      | 3.8290e-05               | 0.001                                        |
| Ubuntu 24.04  | Laptop       | CPU i5-7300HQ | `istft_B` f32 | `hann`      | 1.2795e-05               | 0.003                                        |

---

# STFT-ISTFT-ONNX
将短时傅里叶变换（STFT）或逆短时傅里叶变换（ISTFT）过程导出为 ONNX 格式。

## 介绍
本仓库提供了将自定义 STFT 或 ISTFT 过程导出为 ONNX 格式的工具。请按照 Python 脚本的说明操作，并根据需要配置参数。

## 主要特点
- 将 STFT 和 ISTFT 过程导出为 ONNX 格式。
- 支持音频处理参数的自定义，包括窗口类型、FFT 组件和梅尔频带数量。
- 与 PyTorch 原生 STFT/ISTFT 方法相比，提供高精度结果。

## 以下是使用此库的示例
---
```python
from STFT_Process import STFT_Process
from pydub import AudioSegment
import torch
import soundfile as sf


test_audio = './audio_file.mp3'                     # Input a test.
save_reconstructed_audio = './saved_audio.wav'      # Save the reconstructed.


# 配置参数
MAX_SIGNAL_LENGTH = 1024      # STFT 处理后音频的最大帧数。对于长音频输入，请使用更大的值（例如 4096）。
INPUT_AUDIO_LENGTH = 5120     # 用于静态轴导出的音频输入信号长度（以样本为单位）。最好设置为 NFFT 的整数倍。
WINDOW_TYPE = 'hann'          # STFT 中使用的窗口函数类型。
N_MELS = 100                  # 用于梅尔频谱图的梅尔频带数量。
NFFT = 512                    # STFT 过程中的 FFT 组件数。
HOP_LENGTH = 128              # STFT 中连续帧之间的样本数。
SAMPLE_RATE = 16000           # 目标采样率。
STFT_TYPE = "stft_B"          # stft_A: 仅输出实部；stft_B: 输出实部和虚部
ISTFT_TYPE = "istft_B"        # istft_A: 输入 = [幅度, 相位]; istft_B: 输入 = [幅度, 实部, 虚部]，虚部的数据类型为浮点格式。



# 加载音频
audio = torch.tensor(
    AudioSegment.from_file(test_audio)
    .set_channels(1)
    .set_frame_rate(SAMPLE_RATE)
    .get_array_of_samples(), 
    dtype=torch.float32
)
audio = audio.reshape(1, 1, -1)
audio_parts = audio[:, :, :INPUT_AUDIO_LENGTH]

# 创建 STFT 模型
custom_stft = STFT_Process(
    model_type=STFT_TYPE, 
    n_fft=NFFT, 
    n_mels=N_MELS, 
    hop_len=HOP_LENGTH, 
    max_frames=0,  # Not important here.
    window_type=WINDOW_TYPE
).eval()

# 处理音频（STFT）
real_part, imag_part = custom_stft(audio_parts, 'constant')  # pad_mode 选项：['constant', 'reflect']

# 计算幅值
magnitude = torch.sqrt(real_part**2 + imag_part**2)

# 创建 ISTFT 模型
custom_istft = STFT_Process(
    model_type=ISTFT_TYPE, 
    n_fft=NFFT, 
    n_mels=N_MELS, 
    hop_len=HOP_LENGTH, 
    max_frames=MAX_SIGNAL_LENGTH, 
    window_type=WINDOW_TYPE
).eval()

# 从幅值和相位重建音频
audio_reconstructed = custom_istft(magnitude, real_part, imag_part).to(torch.int16)
sf.write(save_reconstructed_audio, audio_reconstructed[0, 0], SAMPLE_RATE, format='WAVEX')
```
---
