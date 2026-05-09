#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "../../main/c/simd/vector_simd.h"

static uint16_t test_fp32_to_bf16(float s) {
    union {
        float f;
        uint32_t i;
    } u;
    u.f = s;
    if ((u.i & 0x7fffffffU) > 0x7f800000U) {
        return (uint16_t) ((u.i >> 16) | 64U);
    }
    if (!(u.i & 0x7f800000U)) {
        return (uint16_t) ((u.i & 0x80000000U) >> 16);
    }
    return (uint16_t) ((u.i + (0x7fffU + ((u.i >> 16) & 1U))) >> 16);
}

static void assert_close(float actual, float expected, float eps) {
    printf("actual: %f expected: %f eps: %f \n", actual, expected, eps);
    assert(fabsf(actual - expected) <= eps);
}

int main(void) {
    /* Small explicit BF16 samples for correctness smoke tests. */
    uint16_t a_bf16[32] = {
            test_fp32_to_bf16(1.0f),
            test_fp32_to_bf16(2.0f),
            test_fp32_to_bf16(3.0f),
            test_fp32_to_bf16(4.0f),
            test_fp32_to_bf16(0.0f),
            test_fp32_to_bf16(0.0f),
            test_fp32_to_bf16(0.0f),
            test_fp32_to_bf16(0.0f),
            test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f),
test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f),
test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f),
test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f),
test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f),
test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f)
    };
    uint16_t b_bf16[32] = {
            test_fp32_to_bf16(5.0f),
            test_fp32_to_bf16(6.0f),
            test_fp32_to_bf16(7.0f),
            test_fp32_to_bf16(8.0f),
            test_fp32_to_bf16(0.0f),
            test_fp32_to_bf16(0.0f),
            test_fp32_to_bf16(0.0f),
            test_fp32_to_bf16(0.0f),
            test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f),
test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f),
test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f),
test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f),
test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f),
test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f), test_fp32_to_bf16(0.0f)

    };
    float a_f32[32] = {1.0f, 2.0f, 3.0f, 4.0f, 
    0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f
 };

    float out_bf16[1] = {0.0f};
    float out_f32_bf16[1] = {0.0f};

    gemm_bf16(0, (const short *) a_bf16, 0, (const short *) b_bf16, 0,
              NULL, out_bf16, 0, 1, 0, 1, 8, 8, 8, 1);
    gemm_f32_bf16(0, a_f32, 0, (const short *) b_bf16, 0,
                  NULL, out_f32_bf16, 0, 1, 0, 1, 8, 8, 8, 1);

    assert_close(out_bf16[0], 70.0f, 0.5f);
    assert_close(out_f32_bf16[0], 70.0f, 0.5f);
    return 0;
}
