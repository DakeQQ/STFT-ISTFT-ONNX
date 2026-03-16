# STFT-ISTFT-ONNX

Export Conv1d-based Short-Time Fourier Transform (STFT) and Inverse STFT (ISTFT) to ONNX format — no `torch.stft` / `torch.istft` at runtime.

---

## Overview

Standard `torch.stft` and `torch.istft` rely on internal FFT ops that do not export cleanly to ONNX. This project replaces them with equivalent **Conv1d** (for STFT) and **ConvTranspose1d** (for ISTFT) operations, producing portable ONNX graphs that run on any ONNX Runtime backend (CPU, GPU, mobile, web).

The script:

1. Builds a PyTorch `STFT_Process` model using pre-computed DFT basis kernels.
2. Exports the model to ONNX (with optional dynamic axes for variable-length audio).
3. Validates the ONNX output against `torch.stft` / `torch.istft`.
4. Runs a round-trip reconstruction test (STFT → ISTFT).

---

## Model Variants

| Model      | Direction | Inputs                   | Outputs              | Description                                      |
|:----------:|:---------:|:------------------------:|:--------------------:|:-------------------------------------------------|
| `stft_A`   | Forward   | waveform                 | real                 | Cosine (real) projection only                    |
| `stft_B`   | Forward   | waveform                 | real, imag           | Full complex STFT via single Conv1d + split      |
| `istft_A`  | Inverse   | magnitude, phase         | waveform             | Polar-form input → overlap-add reconstruction    |
| `istft_B`  | Inverse   | real, imag               | waveform             | Rectangular-form input → overlap-add reconstruction |

**Recommended pair:** `stft_B` + `istft_B` for lossless round-trip reconstruction.

---

## How It Works

### STFT (Forward Transform)

1. **Center padding** (optional): Pad the input waveform by `NFFT // 2` samples on each side so the first frame is centered on sample 0. Supports `reflect` or `constant` (zero) padding.
2. **Windowed DFT via Conv1d**: A Conv1d layer with fixed (non-trainable) weights applies the windowed DFT basis to the padded signal. The kernel weights are:
   - **Cosine basis**: `cos(2πft/N) · window(t)` for each frequency bin `f`
   - **Sine basis**: `-sin(2πft/N) · window(t)` for each frequency bin `f`
3. **Output**:
   - `stft_A` uses only the cosine kernel → outputs `real` part with shape `(batch, F, T)`.
   - `stft_B` concatenates cosine and sine kernels into one Conv1d, then splits the output along the channel dimension → outputs `(real, imag)`, each `(batch, F, T)`.

Where `F = NFFT // 2 + 1` (one-sided frequency bins) and `T` is the number of frames.

### ISTFT (Inverse Transform)

1. **Prepare input**: Concatenate real and imaginary channels (or convert magnitude + phase to real + imag for `istft_A`).
2. **ConvTranspose1d synthesis**: A transposed convolution with the inverse DFT basis (scaled by `2/N` for one-sided spectrum recovery, windowed) performs overlap-add synthesis.
3. **COLA normalization**: Divide the reconstructed signal by the overlap-summed squared window to satisfy the Constant Overlap-Add (COLA) condition, ensuring perfect reconstruction.
4. **Trim center padding**: If center padding was used during STFT, strip the padded edges from the output.

---

## Configuration Parameters

Edit these at the top of `STFT_Process.py` before running the export:

| Parameter            | Default     | Description                                                                 |
|:---------------------|:-----------:|:----------------------------------------------------------------------------|
| `DYNAMIC_AXES`       | `True`      | `True` → ONNX accepts variable-length audio; `False` → fixed length        |
| `OPSET`              | `17`        | ONNX opset version                                                          |
| `NFFT`               | `400`       | FFT size (number of frequency bins before folding)                          |
| `WIN_LENGTH`         | `400`       | Analysis window length in samples (≤ `NFFT`)                               |
| `HOP_LENGTH`         | `160`       | Hop size between successive frames                                          |
| `WINDOW_TYPE`        | `'hann'`    | Window function: `bartlett`, `blackman`, `hamming`, `hann`, `kaiser`        |
| `CENTER_PAD`         | `True`      | Pad signal so frame centers align with sample indices                       |
| `PAD_MODE`           | `'constant'`| Padding mode when `CENTER_PAD=True`: `'reflect'` or `'constant'`           |
| `INPUT_AUDIO_LENGTH` | `16000`     | Waveform length (samples) for dummy tensors during export                   |
| `MAX_SIGNAL_LENGTH`  | `2048`      | Upper-bound frame count for pre-allocated ISTFT buffers                     |
| `STFT_TYPE`          | `'stft_B'`  | Which STFT variant to export                                                |
| `ISTFT_TYPE`         | `'istft_B'` | Which ISTFT variant to export                                               |

---

## Export ONNX Models

Run the script directly to export and validate:

```bash
python STFT_Process.py
```

This will:
- Export `stft_B.onnx` and `istft_B.onnx` (or whichever variants are configured).
- Print validation results comparing ONNX outputs against `torch.stft` / `torch.istft`.
- Run a round-trip reconstruction test and report the mean absolute error.

**Dependencies:**
```
torch
numpy
onnxruntime
onnxslim
```

---

## Usage as a Library

### STFT → ISTFT Round-Trip (PyTorch)

```python
import torch
from STFT_Process import STFT_Process

# Parameters
NFFT = 400
WIN_LENGTH = 400
HOP_LENGTH = 160
MAX_SIGNAL_LENGTH = 1024
WINDOW_TYPE = 'hann'

# Load audio as a float32 tensor with shape (1, 1, num_samples)
audio = torch.randn(1, 1, 16000)  # Replace with real audio

# Forward STFT
stft_model = STFT_Process(
    model_type='stft_B',
    n_fft=NFFT,
    win_length=WIN_LENGTH,
    hop_len=HOP_LENGTH,
    max_frames=MAX_SIGNAL_LENGTH,
    window_type=WINDOW_TYPE
).eval()

real, imag = stft_model(audio)
# real.shape: (1, NFFT//2+1, T)
# imag.shape: (1, NFFT//2+1, T)

# Inverse STFT
istft_model = STFT_Process(
    model_type='istft_B',
    n_fft=NFFT,
    win_length=WIN_LENGTH,
    hop_len=HOP_LENGTH,
    max_frames=MAX_SIGNAL_LENGTH,
    window_type=WINDOW_TYPE
).eval()

reconstructed = istft_model(real, imag)
# reconstructed.shape: (1, 1, num_samples)
```

### Using the Exported ONNX Models

```python
import numpy as np
import onnxruntime as ort

# Load sessions
stft_sess  = ort.InferenceSession('stft_B.onnx')
istft_sess = ort.InferenceSession('istft_B.onnx')

# Prepare input: float32 array with shape (1, 1, num_samples)
audio = np.random.randn(1, 1, 16000).astype(np.float32)

# Forward STFT
real, imag = stft_sess.run(None, {'input_audio': audio})

# Inverse STFT
reconstructed = istft_sess.run(None, {'real': real, 'imag': imag})[0]
```

### Using `istft_A` (Polar Form)

If you exported with `ISTFT_TYPE = "istft_A"`, the ISTFT model expects magnitude and phase:

```python
# From stft_B outputs (real, imag), convert to polar:
magnitude = np.sqrt(real ** 2 + imag ** 2)
phase     = np.arctan2(imag, real)

reconstructed = istft_sess.run(None, {
    'magnitude': magnitude,
    'phase': phase
})[0]
```

### Full Example with Audio File

```python
from STFT_Process import STFT_Process
from pydub import AudioSegment
import torch
import soundfile as sf

# Configuration
NFFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
SAMPLE_RATE = 16000
MAX_SIGNAL_LENGTH = 1024

# Load audio
audio = torch.tensor(
    AudioSegment.from_file('./audio_file.mp3')
    .set_channels(1)
    .set_frame_rate(SAMPLE_RATE)
    .get_array_of_samples(),
    dtype=torch.float32
).reshape(1, 1, -1)

# STFT
stft = STFT_Process('stft_B', n_fft=NFFT, win_length=WIN_LENGTH, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH).eval()
real, imag = stft(audio)

# ISTFT
istft = STFT_Process('istft_B', n_fft=NFFT, win_length=WIN_LENGTH, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH).eval()
reconstructed = istft(real, imag).to(torch.int16)

sf.write('./reconstructed.wav', reconstructed.reshape(-1), SAMPLE_RATE, format='WAVEX')
```

---

## Tensor Shapes

| Tensor       | Shape                 | Notes                                       |
|:-------------|:----------------------|:--------------------------------------------|
| Input audio  | `(1, 1, num_samples)` | Mono waveform, float32                      |
| STFT real    | `(1, NFFT//2+1, T)`   | `T` = number of frames                     |
| STFT imag    | `(1, NFFT//2+1, T)`   | Same shape as real                          |
| ISTFT output | `(1, 1, num_samples)` | Reconstructed waveform                      |

Frame count `T` depends on padding mode:
- **Center pad** (`CENTER_PAD=True`): `T = num_samples // HOP_LENGTH + 1`
- **No padding** (`CENTER_PAD=False`): `T = (num_samples - NFFT) // HOP_LENGTH + 1`

---

## Window Handling

The window is zero-padded (or center-cropped) to match `NFFT` when `WIN_LENGTH ≠ NFFT`:

- `WIN_LENGTH < NFFT`: Symmetric zero-padding to length `NFFT`.
- `WIN_LENGTH == NFFT`: Used directly.
- `WIN_LENGTH > NFFT`: Center-cropped to length `NFFT`.

Supported window functions: `bartlett`, `blackman`, `hamming`, `hann`, `kaiser` (β=12.0).

---

## Accuracy

Comparison of ONNX model outputs against `torch.stft()` and `torch.istft()`:

| OS            | Device          | Backend        | Model          | Window | Mean Abs. Difference |
|:-------------:|:---------------:|:--------------:|:--------------:|:------:|:--------------------:|
| Ubuntu 24.04  | Laptop i5-7300HQ | CPU           | `stft_A` f32   | hann   | 3.86e-05             |
| Ubuntu 24.04  | Laptop i5-7300HQ | CPU           | `stft_B` f32   | hann   | 3.83e-05             |
| Ubuntu 24.04  | Laptop i5-7300HQ | CPU           | `istft_A` f32  | hann   | 1.35e-05             |
| Ubuntu 24.04  | Laptop i5-7300HQ | CPU           | `istft_B` f32  | hann   | 1.28e-05             |

Real-time factor (chunk size: 160,000 samples / 10 seconds):

| Model      | RTF     |
|:----------:|:-------:|
| `stft_A`   | 0.0005  |
| `stft_B`   | 0.001   |
| `istft_A`  | 0.0015  |
| `istft_B`  | 0.003   |

---

## License

See [LICENSE](LICENSE) for details.


---

# STFT-ISTFT-ONNX

将基于 Conv1d 的短时傅里叶变换（STFT）和逆短时傅里叶变换（ISTFT）导出为 ONNX 格式 — 运行时无需依赖 `torch.stft` / `torch.istft`。

---

## 概述

标准的 `torch.stft` 和 `torch.istft` 依赖内部 FFT 算子，无法干净地导出为 ONNX。本项目使用等价的 **Conv1d**（用于 STFT）和 **ConvTranspose1d**（用于 ISTFT）操作来替代，生成可移植的 ONNX 计算图，可在任意 ONNX Runtime 后端（CPU、GPU、移动端、Web）上运行。

本脚本执行以下步骤：

1. 使用预计算的 DFT 基核构建 PyTorch `STFT_Process` 模型。
2. 将模型导出为 ONNX（可选动态轴，支持可变长度音频）。
3. 将 ONNX 输出与 `torch.stft` / `torch.istft` 进行对比验证。
4. 执行往返重建测试（STFT → ISTFT）。

---

## 模型变体

| 模型       | 方向   | 输入                  | 输出              | 说明                                          |
|:----------:|:------:|:---------------------:|:-----------------:|:----------------------------------------------|
| `stft_A`   | 正变换 | 波形                  | 实部              | 仅余弦（实部）投影                              |
| `stft_B`   | 正变换 | 波形                  | 实部、虚部         | 通过单个 Conv1d + 通道分割实现完整复数 STFT       |
| `istft_A`  | 逆变换 | 幅度、相位            | 波形              | 极坐标输入 → 重叠相加重建                        |
| `istft_B`  | 逆变换 | 实部、虚部            | 波形              | 直角坐标输入 → 重叠相加重建                      |

**推荐组合：** `stft_B` + `istft_B`，可实现无损往返重建。

---

## 算法原理

### STFT（正变换）

1. **中心填充**（可选）：在输入波形两侧各填充 `NFFT // 2` 个采样点，使第一帧的中心对齐采样点 0。支持 `reflect`（反射）和 `constant`（零值）填充模式。
2. **基于 Conv1d 的加窗 DFT**：使用固定权重（不可训练）的 Conv1d 层将加窗 DFT 基应用到填充后的信号上。卷积核权重为：
   - **余弦基**：`cos(2πft/N) · window(t)`，对应每个频率 bin `f`
   - **正弦基**：`-sin(2πft/N) · window(t)`，对应每个频率 bin `f`
3. **输出**：
   - `stft_A` 仅使用余弦核 → 输出实部，形状为 `(batch, F, T)`。
   - `stft_B` 将余弦核和正弦核拼接为一个 Conv1d，然后沿通道维度分割 → 输出 `(real, imag)`，每个形状为 `(batch, F, T)`。

其中 `F = NFFT // 2 + 1`（单侧频率 bin 数），`T` 为帧数。

### ISTFT（逆变换）

1. **准备输入**：拼接实部和虚部通道（对于 `istft_A`，先将幅度 + 相位转换为实部 + 虚部）。
2. **ConvTranspose1d 合成**：使用逆 DFT 基（按 `2/N` 缩放以恢复单侧频谱能量，并加窗）的转置卷积执行重叠相加合成。
3. **COLA 归一化**：将重建信号除以窗函数平方的重叠累加和，满足恒定重叠相加（COLA）条件，确保完美重建。
4. **去除中心填充**：如果 STFT 阶段使用了中心填充，则裁剪输出的填充边缘。

---

## 配置参数

运行导出前，在 `STFT_Process.py` 顶部编辑以下参数：

| 参数                 | 默认值       | 说明                                                                 |
|:---------------------|:-----------:|:---------------------------------------------------------------------|
| `DYNAMIC_AXES`       | `True`      | `True` → ONNX 接受可变长度音频；`False` → 固定长度                    |
| `OPSET`              | `17`        | ONNX opset 版本                                                      |
| `NFFT`               | `400`       | FFT 大小（折叠前的频率 bin 数）                                       |
| `WIN_LENGTH`         | `400`       | 分析窗长度（采样点数，≤ `NFFT`）                                      |
| `HOP_LENGTH`         | `160`       | 连续帧之间的步长                                                      |
| `WINDOW_TYPE`        | `'hann'`    | 窗函数：`bartlett`、`blackman`、`hamming`、`hann`、`kaiser`            |
| `CENTER_PAD`         | `True`      | 填充信号使帧中心与采样点索引对齐                                       |
| `PAD_MODE`           | `'constant'`| `CENTER_PAD=True` 时的填充模式：`'reflect'` 或 `'constant'`           |
| `INPUT_AUDIO_LENGTH` | `16000`     | 导出时虚拟张量的波形长度（采样点数）                                    |
| `MAX_SIGNAL_LENGTH`  | `2048`      | ISTFT 预分配缓冲区的帧数上限                                          |
| `STFT_TYPE`          | `'stft_B'`  | 导出的 STFT 变体                                                      |
| `ISTFT_TYPE`         | `'istft_B'` | 导出的 ISTFT 变体                                                     |

---

## 导出 ONNX 模型

直接运行脚本进行导出和验证：

```bash
python STFT_Process.py
```

脚本将会：
- 导出 `stft_B.onnx` 和 `istft_B.onnx`（或配置的其他变体）。
- 打印 ONNX 输出与 `torch.stft` / `torch.istft` 的对比验证结果。
- 执行往返重建测试并报告平均绝对误差。

**依赖库：**
```
torch
numpy
onnxruntime
onnxslim
```

---

## 作为库使用

### STFT → ISTFT 往返（PyTorch）

```python
import torch
from STFT_Process import STFT_Process

# 参数
NFFT = 400
WIN_LENGTH = 400
HOP_LENGTH = 160
MAX_SIGNAL_LENGTH = 1024
WINDOW_TYPE = 'hann'

# 加载音频为 float32 张量，形状为 (1, 1, 采样点数)
audio = torch.randn(1, 1, 16000)  # 替换为真实音频

# 正变换 STFT
stft_model = STFT_Process(
    model_type='stft_B',
    n_fft=NFFT,
    win_length=WIN_LENGTH,
    hop_len=HOP_LENGTH,
    max_frames=MAX_SIGNAL_LENGTH,
    window_type=WINDOW_TYPE
).eval()

real, imag = stft_model(audio)
# real.shape: (1, NFFT//2+1, T)
# imag.shape: (1, NFFT//2+1, T)

# 逆变换 ISTFT
istft_model = STFT_Process(
    model_type='istft_B',
    n_fft=NFFT,
    win_length=WIN_LENGTH,
    hop_len=HOP_LENGTH,
    max_frames=MAX_SIGNAL_LENGTH,
    window_type=WINDOW_TYPE
).eval()

reconstructed = istft_model(real, imag)
# reconstructed.shape: (1, 1, 采样点数)
```

### 使用导出的 ONNX 模型

```python
import numpy as np
import onnxruntime as ort

# 加载会话
stft_sess  = ort.InferenceSession('stft_B.onnx')
istft_sess = ort.InferenceSession('istft_B.onnx')

# 准备输入：float32 数组，形状为 (1, 1, 采样点数)
audio = np.random.randn(1, 1, 16000).astype(np.float32)

# 正变换 STFT
real, imag = stft_sess.run(None, {'input_audio': audio})

# 逆变换 ISTFT
reconstructed = istft_sess.run(None, {'real': real, 'imag': imag})[0]
```

### 使用 `istft_A`（极坐标形式）

如果导出时设置了 `ISTFT_TYPE = "istft_A"`，ISTFT 模型接受幅度和相位作为输入：

```python
# 从 stft_B 输出的 (real, imag) 转换为极坐标：
magnitude = np.sqrt(real ** 2 + imag ** 2)
phase     = np.arctan2(imag, real)

reconstructed = istft_sess.run(None, {
    'magnitude': magnitude,
    'phase': phase
})[0]
```

### 完整音频文件示例

```python
from STFT_Process import STFT_Process
from pydub import AudioSegment
import torch
import soundfile as sf

# 配置参数
NFFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
SAMPLE_RATE = 16000
MAX_SIGNAL_LENGTH = 1024

# 加载音频
audio = torch.tensor(
    AudioSegment.from_file('./audio_file.mp3')
    .set_channels(1)
    .set_frame_rate(SAMPLE_RATE)
    .get_array_of_samples(),
    dtype=torch.float32
).reshape(1, 1, -1)

# STFT
stft = STFT_Process('stft_B', n_fft=NFFT, win_length=WIN_LENGTH, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH).eval()
real, imag = stft(audio)

# ISTFT
istft = STFT_Process('istft_B', n_fft=NFFT, win_length=WIN_LENGTH, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH).eval()
reconstructed = istft(real, imag).to(torch.int16)

sf.write('./reconstructed.wav', reconstructed.reshape(-1), SAMPLE_RATE, format='WAVEX')
```

---

## 张量形状

| 张量         | 形状                  | 说明                                        |
|:-------------|:--------------------|:--------------------------------------------|
| 输入音频     | `(1, 1, 采样点数)`      | 单声道波形，float32                          |
| STFT 实部    | `(1, NFFT//2+1, T)` | `T` = 帧数                                  |
| STFT 虚部    | `(1, NFFT//2+1, T)` | 与实部形状相同                                |
| ISTFT 输出   | `(1, 1, 采样点数)`      | 重建波形                                     |

帧数 `T` 取决于填充模式：
- **中心填充**（`CENTER_PAD=True`）：`T = 采样点数 // HOP_LENGTH + 1`
- **无填充**（`CENTER_PAD=False`）：`T = (采样点数 - NFFT) // HOP_LENGTH + 1`

---

## 窗函数处理

当 `WIN_LENGTH ≠ NFFT` 时，窗函数会被零填充（或中心裁剪）以匹配 `NFFT`：

- `WIN_LENGTH < NFFT`：对称零填充至长度 `NFFT`。
- `WIN_LENGTH == NFFT`：直接使用。
- `WIN_LENGTH > NFFT`：中心裁剪至长度 `NFFT`。

支持的窗函数：`bartlett`、`blackman`、`hamming`、`hann`、`kaiser`（β=12.0）。

---

## 精度对比

ONNX 模型输出与 `torch.stft()` / `torch.istft()` 的对比：

| 操作系统       | 设备             | 后端           | 模型           | 窗函数 | 平均绝对误差         |
|:-------------:|:---------------:|:--------------:|:--------------:|:------:|:--------------------:|
| Ubuntu 24.04  | 笔记本 i5-7300HQ | CPU           | `stft_A` f32   | hann   | 3.86e-05             |
| Ubuntu 24.04  | 笔记本 i5-7300HQ | CPU           | `stft_B` f32   | hann   | 3.83e-05             |
| Ubuntu 24.04  | 笔记本 i5-7300HQ | CPU           | `istft_A` f32  | hann   | 1.35e-05             |
| Ubuntu 24.04  | 笔记本 i5-7300HQ | CPU           | `istft_B` f32  | hann   | 1.28e-05             |

实时因数（块大小：160,000 采样点 / 10 秒）：

| 模型       | 实时因数 |
|:----------:|:--------:|
| `stft_A`   | 0.0005   |
| `stft_B`   | 0.001    |
| `istft_A`  | 0.0015   |
| `istft_B`  | 0.003    |

---

## 许可证

详见 [LICENSE](LICENSE)。

