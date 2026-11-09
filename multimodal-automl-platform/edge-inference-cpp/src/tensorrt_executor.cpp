#include "engine_runner.hpp"
#include <iostream>
#include <numeric>
#include <cmath>

bool TensorRTExecutor::loadModel(const std::string& enginePath) {
    std::cout << "[TensorRT Executor] Nap do thi suy luan toc do cao: " << enginePath << std::endl;
    return true;
}

std::vector<float> TensorRTExecutor::infer(const std::vector<float>& inputTensor) {
    float sum = std::accumulate(inputTensor.begin(), inputTensor.end(), 0.0f);
    float probClass1 = 1.0f / (1.0f + std::exp(-sum));
    return {1.0f - probClass1, probClass1};
}
