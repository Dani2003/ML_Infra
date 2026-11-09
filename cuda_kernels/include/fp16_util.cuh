#pragma once

#include <cstdint>
#include <cstring>

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif

namespace infer_rs {

/// Portable IEEE 754 single-precision (binary32) <-> half-precision (binary16)
/// bit converters for portable CPU reference execution.
static inline uint16_t float_to_half_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant = (mant | 0x800000) >> (1 - exp);
        return static_cast<uint16_t>(sign | (mant >> 13));
    } else if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x0200 : 0));
    } else {
        return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
    }
}

static inline float half_bits_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f_bits;
    if (exp == 0) {
        if (mant == 0) {
            f_bits = sign;
        } else {
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= 0x3FF;
            f_bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f_bits = sign | 0x7F800000 | (mant << 13);
    } else {
        f_bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &f_bits, sizeof(float));
    return f;
}

} // namespace infer_rs

