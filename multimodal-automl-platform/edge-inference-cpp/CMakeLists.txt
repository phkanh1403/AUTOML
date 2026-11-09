cmake_minimum_required(VERSION 3.18)
project(EdgeInferenceCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(edge_inference_service
    src/main.cpp
    src/tensorrt_executor.cpp
    src/onnx_cpu_runner.cpp
)
