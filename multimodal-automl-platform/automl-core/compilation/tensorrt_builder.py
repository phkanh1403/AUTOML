import os

class TensorRTBuilder:
    @staticmethod
    def build_engine(onnx_path: str, trt_output_path: str, fp16_mode: bool = True) -> str:
        os.makedirs(os.path.dirname(trt_output_path), exist_ok=True)
        with open(trt_output_path, "wb") as f:
            f.write(f"TENSORRT_ENGINE_FP16={fp16_mode}_SOURCE={onnx_path}".encode("utf-8"))
        return trt_output_path
