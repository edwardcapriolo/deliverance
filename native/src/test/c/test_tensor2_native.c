#include "tensor2_native.h"

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
    return 0;
}
