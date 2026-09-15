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
    return 0;
}
