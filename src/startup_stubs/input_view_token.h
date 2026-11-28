// What an input view token resolves to, before a compute or indirect buffer bind.
// nullptr means the token is a constant-buffer view, or a buffer shader-resource view of a
// desktop resource: the caller keeps its range, alignment and alias checks. Every other
// token is refused with the shape that failed, so one census reason is not shared by all of them.
// A non-buffer shader resource view keeps the historical sentence. Placement is the Placement
// enumeration: 0 none, 1 aliased buffer, 2 copied buffer.
#pragma once

#include <cstdint>

struct InputViewTokenFacts {
    bool known = false;
    std::uint32_t slot = 0;
    std::uint32_t desc_bytes = 0;
    std::uint32_t view_dimension = 0;
    bool resource_pointer = false;
    bool wrapper = false;
    bool resource = false;
    bool desktop = false;
    std::uint32_t placement = 0;
};

inline const char* input_view_token_refusal(const InputViewTokenFacts& facts)
{
    constexpr std::uint32_t buffer_dimension = 1;  // D3D12_SRV_DIMENSION_BUFFER
    constexpr std::uint32_t shader_resource = 18;
    constexpr std::uint32_t shader_resource_bytes = 40;
    constexpr std::uint32_t constant_buffer_bytes = 12;
    if (!facts.known) return "an input view token is unknown";
    if (facts.slot == 22) return "an input view token is a sampler";
    if (facts.slot == 19) return "an input view token is an unordered access view";
    if (facts.slot == 20 || facts.slot == 21) return "an input view token is a render or depth view";
    if (facts.slot == 17) {
        if (facts.desc_bytes < constant_buffer_bytes) return "an input view token has a truncated constant buffer view";
        return nullptr;
    }
    if (facts.slot != shader_resource) return "an input view token has an unrecognized slot";
    if (facts.desc_bytes < shader_resource_bytes) return "an input view token has a truncated shader resource view";
    if (facts.view_dimension != buffer_dimension)
        return "an input view token is not a buffer view of a desktop resource";
    if (!facts.resource_pointer || !facts.wrapper) return "an input view token names no resource";
    if (!facts.resource) return "an input view token does not name a resource";
    if (!facts.desktop) {
        if (facts.placement == 1 || facts.placement == 2)
            return "an input view token's placed or copied resource has no desktop backing";
        return "an input view token has no desktop resource";
    }
    return nullptr;
}
