# STFT-ISTFT-ONNX
Export the STFT or ISTFT process in ONNX format.
1. To export your custom STFT or ISTFT process in ONNX format, please follow the instructions provided in the Python script and configure it accordingly.
2. The usage method is as follows:
   ```
   from STFT_Process import STFT_Process
   from pydub import AudioSegment

   MAX_SIGNAL_LENGTH = 1024                # Max frames for audio length after STFT processed. Set a appropriate larger value for long audio input, such as 4096.
   INPUT_AUDIO_LENGTH = 5120               # Set for static axis export: the length of the audio input signal (in samples). It is better to set an integer multiple of the NFFT value.
   WINDOW_TYPE = 'hann'                    # Type of window function used in the STFT
   N_MELS = 100                            # Number of Mel bands to generate in the Mel-spectrogram
   NFFT = 512                              # Number of FFT components for the STFT process
   HOP_LENGTH = 128                        # Number of samples between successive frames in the STFT
   SAMPLE_RATE = 16000                     # The target sample rate.

   test_audio = r'./audio_file.mp3'

   # Load the audio
   audio = torch.tensor(AudioSegment.from_file(test_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=torch.float32) / 32768.0
   audio = audio.reshape(1, 1, -1)
   audio_parts = audio[:, :, :INPUT_AUDIO_LENGTH]

   # Create the stft model, The stft_A output the real_part only.
   custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()

   # Process the audio
   real_part, imag_part = self.stft_model(audio_parts, 'constant')  # pad mode = ['constant', 'reflect'];
   
   #______________________________________________________________________________________________________________________________________________

   # Create the istft model, The istft_A accept magnitude and phase as input.
   custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()

   # Calculate the magnitude
   magnitude = torch.sqrt(real_part * real_part + imag_part * imag_part)

   # Convert it back to the audio.
   audio_parts = self.istft_model(magnitude, real_part, imag_part)
   
   ```



# STFT-ISTFT-ONNX
1. 要将自定义的 STFT 或 ISTFT 处理导出为 ONNX 格式，请按照 Python 脚本中提供的说明进行操作，并进行相应的配置。
2. 使用方法如下：
   ```
   from STFT_Process import STFT_Process
   from pydub import AudioSegment

   MAX_SIGNAL_LENGTH = 1024                # Max frames for audio length after STFT processed. Set a appropriate larger value for long audio input, such as 4096.
   INPUT_AUDIO_LENGTH = 5120               # Set for static axis export: the length of the audio input signal (in samples). It is better to set an integer multiple of the NFFT value.
   WINDOW_TYPE = 'hann'                    # Type of window function used in the STFT
   N_MELS = 100                            # Number of Mel bands to generate in the Mel-spectrogram
   NFFT = 512                              # Number of FFT components for the STFT process
   HOP_LENGTH = 128                        # Number of samples between successive frames in the STFT
   SAMPLE_RATE = 16000                     # The target sample rate.

   test_audio = r'./audio_file.mp3'

   # Load the audio
   audio = torch.tensor(AudioSegment.from_file(test_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=torch.float32) / 32768.0
   audio = audio.reshape(1, 1, -1)
   audio_parts = audio[:, :, :INPUT_AUDIO_LENGTH]

   # Create the stft model, The stft_A output the real_part only.
   custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()

   # Process the audio
   real_part, imag_part = self.stft_model(audio_parts, 'constant')  # pad mode = ['constant', 'reflect'];
   
   #______________________________________________________________________________________________________________________________________________

   # Create the istft model, The istft_A accept magnitude and phase as input.
   custom_istft = STFT_Process(model_type='istft_B', n_fft=NFFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()

   # Calculate the magnitude
   magnitude = torch.sqrt(real_part * real_part + imag_part * imag_part)

   # Convert it back to the audio.
   audio_parts = self.istft_model(magnitude, real_part, imag_part)

   ```

# 差异 Difference<br>Compared to torch.stft() and torch.istft()
| OS | Device | Backend | Model | Window Type | Absloute Mean Difference | Real-Time Factor<br>Chunk_Size: 160000 or 10s |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Ubuntu-24.04 | Laptop | CPU<br>i5-7300HQ | stft_A<br>f32 | hann | 3.8566932e-05 | 0.0005 |
| Ubuntu-24.04 | Laptop | CPU<br>i5-7300HQ | istft_A<br>f32 | hann | 1.348025e-05 | 0.0015 |
| Ubuntu-24.04 | Laptop | CPU<br>i5-7300HQ | stft_B<br>f32 | hann | 3.82898753e-05 | 0.001 |
| Ubuntu-24.04 | Laptop | CPU<br>i5-7300HQ | istft_B<br>f32 | hann | 1.2795282e-05 | 0.003 |
