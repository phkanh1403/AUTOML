#include "engine_runner.hpp"
#include <iostream>
#include <cmath>

bool ONNXCpuRunner::loadModel(const std::string& onnxPath) {
    std::cout << "[ONNX CPU Runner] Khoi tao ONNX Runtime cho thiet bi bien: " << onnxPath << std::endl;
    return true;
}

std::vector<float> ONNXCpuRunner::infer(const std::vector<float>& inputTensor) {
    float score = inputTensor.empty() ? 0.5f : std::tanh(inputTensor[0]);
    return {1.0f - score, score};
}
