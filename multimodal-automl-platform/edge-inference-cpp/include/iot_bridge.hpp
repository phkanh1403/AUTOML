#ifndef IOT_BRIDGE_HPP
#define IOT_BRIDGE_HPP

#include <vector>
#include <string>

struct SensorFrame {
    std::string deviceId;
    std::vector<float> features;
};

class IoTBridge {
public:
    static SensorFrame readEmbeddedFrame(const std::string& deviceId) {
        return SensorFrame{deviceId, {0.42f, -0.18f, 1.05f, 0.88f}};
    }
};

#endif
