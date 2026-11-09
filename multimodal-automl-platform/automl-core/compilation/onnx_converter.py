import torch
import torch.nn as nn
import os

class ONNXCompiler:
    @staticmethod
    def export_pytorch_to_onnx(model: nn.Module, output_onnx_path: str, input_shape=(1, 3, 224, 224)) -> str:
        os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
        model.eval()
        dummy_input = torch.randn(*input_shape)
        torch.onnx.export(
            model, dummy_input, output_onnx_path, opset_version=17,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        return output_onnx_path
