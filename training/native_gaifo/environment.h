#pragma once
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>

// CPU pointers are contiguous, actor-major within each arena. Owns no Torch objects.
class Environments {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    Environments(const nlohmann::json& config, const float* resets, const float* internals,
                 int64_t resetCount);
    ~Environments();
    void Observe(float* output);
    void Step(const int64_t* actions, float* next, float* final,
              uint8_t* done, uint8_t* terminated, float* scores);
};
