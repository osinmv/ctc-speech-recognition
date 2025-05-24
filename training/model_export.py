import torch
import os
from model import MyMediumLSTMModel
def export_to_onnx(model, onnx_model_path, input_shape):

    # Create a dummy input tensor
    dummy_input = torch.randn(input_shape)

    # Export the model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,  # Store the trained parameter weights inside the ONNX file
        opset_version=11,  # Choose an appropriate ONNX opset version
        do_constant_folding=True,  # Optimize constants
        input_names=['input'],  # Assign names to input nodes
        output_names=['output'],  # Assign names to output nodes
        dynamic_axes={'input': {0: 'batch_size', 1:"sequence_length"}, 'output': {0: 'batch_size', 1:"sequence_length"}} # Allow dynamic batch size
    )

    print(f"Model exported to {onnx_model_path}")


if __name__ == '__main__':
    # Example usage:
    model = MyMediumLSTMModel.eval()
    model.load_state_dict(torch.load(os.getenv("CHECKPOINT","model/model-MediumLSTMModel-checkpoint-9.pth")))

    onnx_model_path = 'CurrentlyBest/model_medium.onnx'
    input_shape = (1, 128, 13)
    export_to_onnx(model, onnx_model_path, input_shape)