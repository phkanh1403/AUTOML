#include "engine_runner.hpp"
#include "iot_bridge.hpp"
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
    std::cout << "=== KHOI CHAY LOI SUY LUAN BIEN C++ (ONNX / TensorRT) ===" << std::endl;
    std::unique_ptr<IEngineRunner> runner = std::make_unique<TensorRTExecutor>();
    runner->loadModel("../data/artifacts/model_latest.trt");

    SensorFrame frame = IoTBridge::readEmbeddedFrame("ESP32_EDGE_NODE_01");
    std::vector<float> probs = runner->infer(frame.features);

    std::cout << "[Edge Inference] Thiet bi: " << frame.deviceId
              << " | Xac suat lop [0, 1]: [" << probs[0] << ", " << probs[1] << "]" << std::endl;
    return 0;
}
