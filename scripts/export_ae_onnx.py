import torch
import os
from pathlib import Path
from src.ae_gate import _ConvAutoencoder, AutoencoderWithMSE

def export_ae_onnx(output_path="models/autoencoder_mse.onnx"):
    print(f"Exporting Autoencoder with MSE to {output_path}...")
    
    # 1. Initialize model
    base_model = _ConvAutoencoder()
    model = AutoencoderWithMSE(base_model)
    model.eval()

    # 2. Create dummy input [Batch, Channels, Height, Width]
    # DeepStream will resize to 128x128 as defined in nvinfer config
    dummy_input = torch.randn(1, 3, 128, 128)

    # 3. Export
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['mse_score'],
        dynamic_axes={'input': {0: 'batch_size'}, 'mse_score': {0: 'batch_size'}}
    )
    print("Export complete.")

if __name__ == "__main__":
    export_ae_onnx()
