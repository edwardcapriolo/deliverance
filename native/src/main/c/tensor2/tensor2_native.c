#include "tensor2_native.h"

#define TENSOR2_Q8_BLOCK_SIZE 32

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

static inline float tensor2_dot_f32_q8(
        const float *a,
        const int8_t *b,
        const float *b_scales,
        int column_length) {
    int block_count = column_length / TENSOR2_Q8_BLOCK_SIZE;
#if defined(TENSOR2_ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int block = 0; block < block_count; block++) {
        float32x4_t scale = vdupq_n_f32(b_scales[block]);
        int base = block * TENSOR2_Q8_BLOCK_SIZE;
        for (int offset = 0; offset < TENSOR2_Q8_BLOCK_SIZE; offset += 16) {
            int8x16_t raw = vld1q_s8(b + base + offset);
            int16x8_t lo16 = vmovl_s8(vget_low_s8(raw));
            int16x8_t hi16 = vmovl_s8(vget_high_s8(raw));
            float32x4_t q0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), scale);
            float32x4_t q1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), scale);
            float32x4_t q2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), scale);
            float32x4_t q3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), scale);
            acc = vmlaq_f32(acc, vld1q_f32(a + base + offset), q0);
            acc = vmlaq_f32(acc, vld1q_f32(a + base + offset + 4), q1);
            acc = vmlaq_f32(acc, vld1q_f32(a + base + offset + 8), q2);
            acc = vmlaq_f32(acc, vld1q_f32(a + base + offset + 12), q3);
        }
    }
    return tensor2_hsum4(acc);
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (int block = 0; block < block_count; block++) {
        __m256 scale = _mm256_set1_ps(b_scales[block]);
        int base = block * TENSOR2_Q8_BLOCK_SIZE;
        for (int offset = 0; offset < TENSOR2_Q8_BLOCK_SIZE; offset += 8) {
            __m128i raw8 = _mm_loadl_epi64((const __m128i *) (b + base + offset));
            __m256 q = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw8)), scale);
            __m256 av = _mm256_loadu_ps(a + base + offset);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(av, q));
        }
    }
    return tensor2_hsum8(acc);
#else
    return 0.0f;
#endif
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

tensor2_status tensor2_batch_dot_f32_q8(
        float *result,
        const float *a,
        const int8_t *b,
        const float *b_scales,
        int result_rows,
        int b_rows,
        int column_length,
        int result_stride,
        int a_stride,
        int b_stride,
        int b_scale_stride) {
    if (result == 0 || a == 0 || b == 0 || b_scales == 0) {
        return TENSOR2_UNSUPPORTED;
    }
    if (result_rows < 0 || b_rows < 0 || column_length < 0
            || result_stride < 0 || a_stride < 0 || b_stride < 0 || b_scale_stride < 0) {
        return TENSOR2_UNSUPPORTED;
    }
    if (column_length % TENSOR2_Q8_BLOCK_SIZE != 0) {
        return TENSOR2_UNSUPPORTED;
    }
#if !defined(TENSOR2_ARM_NEON) && !defined(__AVX2__)
    return TENSOR2_UNSUPPORTED;
#else
    int blocks = column_length / TENSOR2_Q8_BLOCK_SIZE;
    for (int result_row = 0; result_row < result_rows; result_row++) {
        const float *a_row = a + result_row * a_stride;
        for (int b_row = 0; b_row < b_rows; b_row++) {
            const int8_t *b_row_ptr = b + b_row * b_stride;
            const float *scale_row = b_scales + b_row * b_scale_stride;
            result[result_row * result_stride + b_row] =
                    tensor2_dot_f32_q8(a_row, b_row_ptr, scale_row, blocks * TENSOR2_Q8_BLOCK_SIZE);
        }
    }
    return TENSOR2_OK;
#endif
}
