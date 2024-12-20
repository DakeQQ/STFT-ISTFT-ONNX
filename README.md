# STFT-ISTFT-ONNX
Export the STFT or ISTFT process in ONNX format.
1. To export your custom STFT or ISTFT process in ONNX format, please follow the instructions provided in the Python script and configure it accordingly.
2. The usage method is as follows:
   ```
   from STFT_Process import STFT_Process

   # Create the stft model, The stft_A output the real_part only.
   custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()

   # Process the audio
   real_part, imag_part = self.stft_model(audio, 'constant')  # pad mode = ['constant', 'reflect'];
   
   #______________________________________________________________________________________________________________________________________________

   # Create the istft model, The istft_A accept magnitude and phase as input.
   custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()

   # Calculate the magnitude
   magnitude = torch.sqrt(real_part * real_part + imag_part * imag_part)

   # Convert it back to the audio.
   audio = self.istft_model(magnitude, real_part, imag_part)
   
   ```



# STFT-ISTFT-ONNX
1. 要将自定义的 STFT 或 ISTFT 处理导出为 ONNX 格式，请按照 Python 脚本中提供的说明进行操作，并进行相应的配置。
2. 使用方法如下：
   ```
   from STFT_Process import STFT_Process

   # Create the stft model, The stft_A output the real_part only.
   custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()

   # Process the audio
   real_part, imag_part = self.stft_model(audio, 'constant')  # pad mode = ['constant', 'reflect'];
   
   #______________________________________________________________________________________________________________________________________________

   # Create the istft model, The istft_A accept magnitude and phase as input.
   custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()

   # Calculate the magnitude
   magnitude = torch.sqrt(real_part * real_part + imag_part * imag_part)

   # Convert it back to the audio.
   audio = self.istft_model(magnitude, real_part, imag_part)
   
   ```

# 差异 Difference - Compared to torch.stft() and torch.istft()
| OS | Device | Backend | Model | Window Type | Absloute Mean Difference |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Ubuntu-24.04 | Desktop | CPU<br>i3-12300 | stft_A | hann | 3.8566932e-05 |
| Ubuntu-24.04 | Desktop | CPU<br>i3-12300 | istft_A | hann | 1.348025e-05 |
| Ubuntu-24.04 | Desktop | CPU<br>i3-12300 | stft_B | hann | 3.82898753e-05 |
| Ubuntu-24.04 | Desktop | CPU<br>i3-12300 | istft_B | hann | 1.2795282e-05 |
