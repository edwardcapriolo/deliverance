#include "tensor2_native.h"

static float bf16_to_f32(uint16_t value) {
    union {
        uint32_t bits;
        float f;
    } converted;
    converted.bits = ((uint32_t) value) << 16;
    return converted.f;
}

static uint16_t f32_to_bf16(float value) {
    union {
        uint32_t bits;
        float f;
    } converted;
    converted.f = value;
    uint32_t lsb = (converted.bits >> 16) & 1u;
    converted.bits += 0x7fffu + lsb;
    return (uint16_t) (converted.bits >> 16);
}

int main(void) {
    float input[4 * 4] = {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f
    };
    float weight[3 * 4] = {
            1.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f
    };
    float result[2 * 5] = {0};

    tensor2_status status = tensor2_batch_dot_f32_f32(
            result,
            input,
            weight,
            2,
            1,
            0,
            0,
            4,
            1,
            0,
            3,
            5,
            4,
            4);
    if (status != TENSOR2_OK) {
        return 2;
    }
    if (result[1] != 12.0f || result[2] != 14.0f || result[3] != 26.0f) {
        return 3;
    }
    if (result[6] != 20.0f || result[7] != 22.0f || result[8] != 42.0f) {
        return 4;
    }
    float q8_a[32] = {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
            17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
            25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f
    };
    signed char q8_b[2 * 32] = {0};
    for (int i = 0; i < 32; i++) {
        q8_b[i] = 1;
        q8_b[32 + i] = 2;
    }
    float q8_scales[2] = {0.5f, 0.25f};
    float q8_result[2] = {0};
    status = tensor2_batch_dot_f32_q8(q8_result, q8_a, q8_b, q8_scales,
            1, 2, 32, 2, 32, 32, 1);
    if (status != TENSOR2_OK) {
        return 5;
    }
    if (q8_result[0] != 264.0f || q8_result[1] != 264.0f) {
        return 6;
    }
    float scale_target[2 * 5] = {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            6.0f, 7.0f, 8.0f, 9.0f, 10.0f
    };
    status = tensor2_scale_f32(scale_target, 2.0f, 2, 1, 3, 5);
    if (status != TENSOR2_OK) {
        return 7;
    }
    if (scale_target[0] != 1.0f || scale_target[1] != 4.0f || scale_target[2] != 6.0f
            || scale_target[3] != 8.0f || scale_target[4] != 5.0f) {
        return 8;
    }
    if (scale_target[5] != 6.0f || scale_target[6] != 14.0f || scale_target[7] != 16.0f
            || scale_target[8] != 18.0f || scale_target[9] != 10.0f) {
        return 9;
    }
    uint16_t bf16_target[2 * 9];
    for (int i = 0; i < 18; i++) {
        bf16_target[i] = f32_to_bf16((float) (i + 1));
    }
    status = tensor2_scale_bf16(bf16_target, -1.5f, 2, 2, 5, 9);
    if (status != TENSOR2_OK) {
        return 10;
    }
    for (int row = 0; row < 2; row++) {
        for (int column = 0; column < 9; column++) {
            float expected = (float) (row * 9 + column + 1);
            if (column >= 2 && column < 7) {
                expected *= -1.5f;
            }
            float actual = bf16_to_f32(bf16_target[row * 9 + column]);
            float diff = actual > expected ? actual - expected : expected - actual;
            if (diff > 0.05f) {
                return 11;
            }
        }
    }
    return 0;
}
