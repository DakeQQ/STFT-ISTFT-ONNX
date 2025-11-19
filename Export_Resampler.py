import torch
import onnx
import onnxruntime
import numpy as np
from onnxslim import slim

# ==========================================
# 1. Configuration
# ==========================================

DYNAMIC = False              # Set True for the dynamic shape.
USE_MEAN_POOLING = True      # This will trigger the Reshape + Mean optimization. Only works for integer downsample case.
INPUT_LENGTH = 48000         # Set for the static shape.
RESAMPLE_RATIO = 1/3         # 1/3 means one-third of the input length.
INPUT_DTYPE = torch.int16    # Options: torch.float32 or torch.int16
OUTPUT_DTYPE = torch.int16   # Options: torch.float32 or torch.int16
save_path = f"resampler_{INPUT_LENGTH}_to_{int(INPUT_LENGTH * RESAMPLE_RATIO)}.onnx"


# ==========================================
# 2. Define the Model
# ==========================================
class ResamplerModel(torch.nn.Module):
    def __init__(self, scale_factor):
        super(ResamplerModel, self).__init__()
        self.scale_factor = scale_factor

        # Determine if this is an integer downsample case
        # We calculate the inverse (e.g., 0.25 -> 4.0)
        inverse_scale = 1.0 / scale_factor

        # Check if scale < 1 (downsampling) AND inverse is effectively an integer
        is_downsample = scale_factor < 1.0
        is_integer_factor = abs(inverse_scale - round(inverse_scale)) < 1e-6

        self.use_mean_pooling = is_downsample and is_integer_factor and USE_MEAN_POOLING

        if self.use_mean_pooling:
            self.stride = int(round(inverse_scale))
            print(f"[Model Init] Optimized mode: Reshape + Mean (Factor: {self.stride})")
        else:
            print(f"[Model Init] Standard mode: Interpolate (Scale: {self.scale_factor})")

    def forward(self, x):
        if self.use_mean_pooling:
            x = x.reshape(1, 1, -1, self.stride)
            x = x.to(torch.float32)
            out = torch.sum(x, dim=-1, keepdim=False) * self.scale_factor
        else:
            if INPUT_DTYPE == torch.int16:
                x = x.float()
            out = torch.nn.functional.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode='linear',
                align_corners=False,
                recompute_scale_factor=True
            )
        out = out.clamp(min=-32768.0, max=32767.0)
        if OUTPUT_DTYPE == torch.int16:
            out = out.to(torch.int16)
        return out


# ==========================================
# 3. Export Function
# ==========================================
def export_to_onnx():
    print(f"\n--- Starting Export (Ratio: {RESAMPLE_RATIO}) ---")

    model = ResamplerModel(scale_factor=RESAMPLE_RATIO)
    model.eval()

    dummy_input = torch.ones(1, 1, INPUT_LENGTH, dtype=INPUT_DTYPE)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_audio'],
        output_names=['output_audio'],
        dynamic_axes={
        'input_audio': {2: 'audio_len'},
        'output_audio': {2: 'resampled_len'}
        } if DYNAMIC else None,
        dynamo=False
    )

    slim(
        model=save_path,
        output_model=save_path,
        no_shape_infer=False,          # False for more optimize but may get errors.
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=False,
        verbose=False
    )

    print(f"Model exported to: {save_path}")
    return save_path


# ==========================================
# 4. Verification
# ==========================================
def verify_onnx(onnx_path):
    print("\n--- Verifying with ONNX Runtime ---")

    ort_session = onnxruntime.InferenceSession(onnx_path)

    if INPUT_DTYPE == torch.float32:
        ort_input_data = np.random.randn(1, 1, INPUT_LENGTH).astype(np.float32)
    else:
        ort_input_data = np.random.randint(-32768, 32767, (1, 1, INPUT_LENGTH)).astype(np.int16)

    ort_inputs = {ort_session.get_inputs()[0].name: ort_input_data}
    ort_outs = ort_session.run(None, ort_inputs)
    output_tensor = ort_outs[0]

    print(f"Input Shape:  {ort_input_data.shape}")
    print(f"Output Shape: {output_tensor.shape}")

    # Validation Logic
    expected_len = int(INPUT_LENGTH * RESAMPLE_RATIO)
    actual_len = output_tensor.shape[2]

    print(f"Expected Len: {expected_len}")
    print(f"Actual Len:   {actual_len}")

    # Note: If using reshape/mean, we truncate.
    # If using interpolate, it might round differently.
    # We allow a difference of 1 sample due to truncation vs interpolation rounding.
    if abs(actual_len - expected_len) <= 1:
        print("SUCCESS: Output length is correct.")
    else:
        print("WARNING: Output length mismatch.")


if __name__ == "__main__":
    saved_path = export_to_onnx()
    verify_onnx(saved_path)
