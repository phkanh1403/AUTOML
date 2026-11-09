#ifndef ENGINE_RUNNER_HPP
#define ENGINE_RUNNER_HPP

#include <string>
#include <vector>

class IEngineRunner {
public:
    virtual ~IEngineRunner() = default;
    virtual bool loadModel(const std::string& modelPath) = 0;
    virtual std::vector<float> infer(const std::vector<float>& inputTensor) = 0;
};

class TensorRTExecutor : public IEngineRunner {
public:
    bool loadModel(const std::string& enginePath) override;
    std::vector<float> infer(const std::vector<float>& inputTensor) override;
};

class ONNXCpuRunner : public IEngineRunner {
public:
    bool loadModel(const std::string& onnxPath) override;
    std::vector<float> infer(const std::vector<float>& inputTensor) override;
};

#endif
