import torch
import torch.nn as nn
from onnxslim import slim

# Export to ONNX
DYANMIC = True                          # Set True for dynamic shape.
INPUT_AUDIO_LENGTH = 16000              # Set for static shape.
DTYPE = torch.int16                     # [torch.float32, torch.int16]
save_path = "normalize_to_int16.onnx"

class NormalizeToInt16(nn.Module):
    def __init__(self):
        super(NormalizeToInt16, self).__init__()
        if DTYPE != torch.int16:
            self.eps = torch.tensor([1e-6], dtype=DTYPE)
        else:
            self.eps = torch.tensor([1], dtype=DTYPE)
        self.eps = self.eps.view(1, 1, -1)

    def forward(self, audio):
        max_val, _ = torch.max(torch.abs(audio), dim=-1, keepdim=True)
        scaling_factor = (32767.0 / (max_val + self.eps))
        if DTYPE == torch.int16:
            scaling_factor = scaling_factor.to(torch.int16)
        normalized = audio * scaling_factor
        normalized = torch.clamp(normalized, -32768, 32767)
        return normalized


def export_to_onnx():
    model = NormalizeToInt16()
    model.eval()
    dummy_input = torch.ones(1, 1, INPUT_AUDIO_LENGTH, dtype=DTYPE)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['audio'],
        output_names=['normalized_audio'],
        dynamic_axes={
            'audio': {2: 'audio_length'},
            'normalized': {2: 'audio_length'}
        } if DYANMIC else None,
        dynamo=False
    )

    slim(
        model=save_path,
        output_model=save_path,
        no_shape_infer=False,            # False for more optimize but may get errors.
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=False,
        verbose=False
    )

    print(f"Model exported successfully to {save_path}")


if __name__ == "__main__":
    export_to_onnx()
