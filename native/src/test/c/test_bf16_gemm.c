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

static float test_bf16_to_fp32(uint16_t s) {
    union {
        uint32_t i;
        float f;
    } u;
    u.i = ((uint32_t) s) << 16;
    return u.f;
}

static float deterministic_value(int seed) {
    return ((seed * 37) % 257 - 128) / 32.0f;
}

static int deterministic_q4(int row, int col) {
    return ((row * 17 + col * 5) % 15) - 7;
}

static int q4_byte_index(int row, int col, int ldb) {
    return row * ldb + (col / Q4_BLOCK_SIZE) * (Q4_BLOCK_SIZE / 2) + (col % (Q4_BLOCK_SIZE / 2));
}

static void set_q4(char *b, int row, int col, int ldb, int q) {
    int byte_index = q4_byte_index(row, col, ldb);
    unsigned char current = (unsigned char) b[byte_index];
    unsigned char nibble = (unsigned char) ((q + 8) & 0x0f);
    if ((col % Q4_BLOCK_SIZE) < (Q4_BLOCK_SIZE / 2)) {
        current = (unsigned char) ((current & 0xf0) | nibble);
    } else {
        current = (unsigned char) ((current & 0x0f) | (nibble << 4));
    }
    b[byte_index] = (char) current;
}

static int get_q4(const char *b, int row, int col, int ldb) {
    unsigned char packed = (unsigned char) b[q4_byte_index(row, col, ldb)];
    int nibble = (col % Q4_BLOCK_SIZE) < (Q4_BLOCK_SIZE / 2) ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
    return nibble - 8;
}

static void scalar_bf16_q4_reference(const uint16_t *a, const float *bf, const char *b, float *out,
                                     int aoffset, int boffset, int m, int n0, int n, int k,
                                     int lda, int ldb, int ldbf, int ldc) {
    for (int row = 0; row < m; row++) {
        for (int out_col = 0; out_col < n; out_col++) {
            int weight_row = n0 + out_col;
            float sum = 0.0f;
            for (int col = 0; col < k; col++) {
                int logical_col = col;
                float av = test_bf16_to_fp32(a[row * lda + aoffset + logical_col]);
                int block = (boffset + logical_col) / Q4_BLOCK_SIZE;
                int q = get_q4(b, weight_row, boffset + logical_col, ldb);
                float scale = bf[weight_row * ldbf + block];
                sum += av * q * scale;
            }
            out[row * ldc + out_col] = sum;
        }
    }
}

static void test_bf16_q4_case(int m, int n, int k, int aoffset, int boffset) {
    int lda = k + aoffset + 16;
    int ldb = (k + boffset + 16) / 2;
    int ldbf = (k + boffset + 16) / Q4_BLOCK_SIZE;
    int ldc = n;
    uint16_t a[m * lda];
    float bf[n * ldbf];
    char b[n * ldb];
    float expected[m * n];
    float actual[m * n];

    for (int i = 0; i < m * lda; i++) {
        a[i] = test_fp32_to_bf16(deterministic_value(i + 3));
    }
    for (int i = 0; i < n * ldbf; i++) {
        bf[i] = 0.5f + (float) ((i * 13) % 7) / 8.0f;
    }
    for (int i = 0; i < n * ldb; i++) {
        b[i] = 0;
    }
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < ldb * 2; col++) {
            set_q4(b, row, col, ldb, deterministic_q4(row, col));
        }
    }
    for (int i = 0; i < m * n; i++) {
        expected[i] = 0.0f;
        actual[i] = 0.0f;
    }

    scalar_bf16_q4_reference(a, bf, b, expected, aoffset, boffset, m, 0, n, k, lda, ldb, ldbf, ldc);
    gemm_bf16_q4(0, (const short *) a, aoffset, bf, b, boffset / 2, actual, 0, m, 0, n, k, lda, ldb, ldbf, ldc);

    for (int i = 0; i < m * n; i++) {
        assert_close(actual[i], expected[i], 0.01f);
    }
}

static void test_bf16_q4_fuzz_cases(void) {
    test_bf16_q4_case(1, 1, 32, 0, 0);
    test_bf16_q4_case(3, 4, 256, 0, 0);
    test_bf16_q4_case(2, 5, 128, 32, 32);
    test_bf16_q4_case(13, 17, 256, 0, 0);
}

static void test_f32_q4_tail_corner_is_written(void) {
    enum { m = 13, n = 16, k = 128 };
    float a[m * k];
    float bf[n * (k / Q4_BLOCK_SIZE)];
    char b[n * (k / 2)];
    float out[m * n];

    for (int i = 0; i < m * k; i++) {
        a[i] = 1.0f;
    }
    for (int i = 0; i < n * (k / Q4_BLOCK_SIZE); i++) {
        bf[i] = 1.0f;
    }
    for (int i = 0; i < n * (k / 2); i++) {
        b[i] = (char) 0x99; /* both Q4 nibbles decode to 1: 9 - 8 */
    }
    for (int i = 0; i < m * n; i++) {
        out[i] = 0.0f;
    }

    gemm_f32_q4(0, a, 0, bf, b, 0, out, 0, m, 0, n, k, k, k / 2, k / 32, n);

    /* Before the bottom-right recursion fix, row 10 col 15 was never written. */
    assert_close(out[10 * n + 15], 128.0f, 0.001f);
    assert_close(out[12 * n + 15], 128.0f, 0.001f);
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
    test_f32_q4_tail_corner_is_written();
    test_bf16_q4_fuzz_cases();
    return 0;
}
