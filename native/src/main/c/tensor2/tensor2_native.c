#include "tensor2_native.h"

#if defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM64)
#define TENSOR2_ARM_NEON 1
#endif

#if defined(TENSOR2_ARM_NEON)
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif

#if defined(TENSOR2_ARM_NEON)
static inline float tensor2_hsum4(float32x4_t v) {
    return vaddvq_f32(v);
}
#endif

#if !defined(TENSOR2_ARM_NEON)
static inline float tensor2_hsum8(__m256 v) {
    float values[8] __attribute__((aligned(32)));
    _mm256_store_ps(values, v);
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        sum += values[i];
    }
    return sum;
}

#if defined(__AVX512F__)
static inline float tensor2_hsum16(__m512 v) {
    return _mm512_reduce_add_ps(v);
}
#endif
#endif

static inline float tensor2_dot_f32_f32(
        const float *a,
        const float *b,
        int a_offset,
        int b_offset,
        int column_length) {
    int k = 0;
    float sum = 0.0f;
#if defined(TENSOR2_ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int upper = (column_length / 4) * 4;
    for (; k < upper; k += 4) {
        float32x4_t av = vld1q_f32(a + a_offset + k);
        float32x4_t bv = vld1q_f32(b + b_offset + k);
        acc = vmlaq_f32(acc, av, bv);
    }
    sum = tensor2_hsum4(acc);
#else
#if defined(__AVX512F__)
    __m512 acc512 = _mm512_setzero_ps();
    int upper512 = (column_length / 16) * 16;
    for (; k < upper512; k += 16) {
        __m512 av = _mm512_loadu_ps(a + a_offset + k);
        __m512 bv = _mm512_loadu_ps(b + b_offset + k);
        acc512 = _mm512_fmadd_ps(av, bv, acc512);
    }
    sum = tensor2_hsum16(acc512);
#else
    __m256 acc256 = _mm256_setzero_ps();
    int upper256 = (column_length / 8) * 8;
    for (; k < upper256; k += 8) {
        __m256 av = _mm256_loadu_ps(a + a_offset + k);
        __m256 bv = _mm256_loadu_ps(b + b_offset + k);
        acc256 = _mm256_fmadd_ps(av, bv, acc256);
    }
    sum = tensor2_hsum8(acc256);
#endif
#endif
    for (; k < column_length; k++) {
        sum += a[a_offset + k] * b[b_offset + k];
    }
    return sum;
}

tensor2_status tensor2_batch_dot_f32_f32(
        float *result,
        const float *a,
        const float *b,
        int result_rows,
        int a_row_offset,
        int a_column_offset,
        int b_column_offset,
        int column_length,
        int result_row_offset,
        int b_row_offset,
        int row_chunk_size,
        int result_stride,
        int a_stride,
        int b_stride) {
    if (result == 0 || a == 0 || b == 0) {
        return TENSOR2_UNSUPPORTED;
    }
    if (result_rows < 0 || a_row_offset < 0 || a_column_offset < 0 || b_column_offset < 0
            || column_length < 0 || result_row_offset < 0 || b_row_offset < 0 || row_chunk_size < 0
            || result_stride < 0 || a_stride < 0 || b_stride < 0) {
        return TENSOR2_UNSUPPORTED;
    }

    for (int result_row = 0; result_row < result_rows; result_row++) {
        int a_offset = (a_row_offset + result_row) * a_stride + a_column_offset;
        for (int b_row_delta = 0; b_row_delta < row_chunk_size; b_row_delta++) {
            int b_row = b_row_offset + b_row_delta;
            int b_offset = b_row * b_stride + b_column_offset;
            result[result_row * result_stride + result_row_offset + b_row] =
                    tensor2_dot_f32_f32(a, b, a_offset, b_offset, column_length);
        }
    }
    return TENSOR2_OK;
}
