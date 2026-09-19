#include "tensor2_native.h"

#include <string.h>

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

static inline float tensor2_bf16_to_f32(uint16_t value) {
    uint32_t bits = ((uint32_t) value) << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static inline uint16_t tensor2_f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (uint16_t) (bits >> 16);
}

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

tensor2_status tensor2_scale_f32(
        float *target,
        float factor,
        int rows,
        int offset,
        int length,
        int stride) {
    if (target == 0) {
        return TENSOR2_UNSUPPORTED;
    }
    if (rows < 0 || offset < 0 || length < 0 || stride < 0) {
        return TENSOR2_UNSUPPORTED;
    }
#if defined(TENSOR2_ARM_NEON)
    float32x4_t scale = vdupq_n_f32(factor);
    for (int row = 0; row < rows; row++) {
        float *row_ptr = target + row * stride + offset;
        int column = 0;
        int upper = (length / 4) * 4;
        for (; column < upper; column += 4) {
            float32x4_t values = vld1q_f32(row_ptr + column);
            vst1q_f32(row_ptr + column, vmulq_f32(values, scale));
        }
        for (; column < length; column++) {
            row_ptr[column] *= factor;
        }
    }
    return TENSOR2_OK;
#elif defined(__AVX2__)
    __m256 scale = _mm256_set1_ps(factor);
    for (int row = 0; row < rows; row++) {
        float *row_ptr = target + row * stride + offset;
        int column = 0;
#if defined(__AVX512F__)
        __m512 scale512 = _mm512_set1_ps(factor);
        int upper512 = (length / 16) * 16;
        for (; column < upper512; column += 16) {
            __m512 values = _mm512_loadu_ps(row_ptr + column);
            _mm512_storeu_ps(row_ptr + column, _mm512_mul_ps(values, scale512));
        }
#endif
        int upper256 = (length / 8) * 8;
        for (; column < upper256; column += 8) {
            __m256 values = _mm256_loadu_ps(row_ptr + column);
            _mm256_storeu_ps(row_ptr + column, _mm256_mul_ps(values, scale));
        }
        for (; column < length; column++) {
            row_ptr[column] *= factor;
        }
    }
    return TENSOR2_OK;
#else
    return TENSOR2_UNSUPPORTED;
#endif
}

tensor2_status tensor2_scale_bf16(
        uint16_t *target,
        float factor,
        int rows,
        int offset,
        int length,
        int stride) {
    if (target == 0) {
        return TENSOR2_UNSUPPORTED;
    }
    if (rows < 0 || offset < 0 || length < 0 || stride < 0) {
        return TENSOR2_UNSUPPORTED;
    }
#if defined(TENSOR2_ARM_NEON)
    float32x4_t scale = vdupq_n_f32(factor);
    for (int row = 0; row < rows; row++) {
        uint16_t *row_ptr = target + row * stride + offset;
        int column = 0;
        int upper = (length / 8) * 8;
        for (; column < upper; column += 8) {
            uint16x8_t raw = vld1q_u16(row_ptr + column);
            uint32x4_t lo_bits = vshll_n_u16(vget_low_u16(raw), 16);
            uint32x4_t hi_bits = vshll_n_u16(vget_high_u16(raw), 16);
            float32x4_t lo = vreinterpretq_f32_u32(lo_bits);
            float32x4_t hi = vreinterpretq_f32_u32(hi_bits);
            uint32x4_t lo_scaled = vreinterpretq_u32_f32(vmulq_f32(lo, scale));
            uint32x4_t hi_scaled = vreinterpretq_u32_f32(vmulq_f32(hi, scale));
            uint32x4_t lo_lsb = vandq_u32(vshrq_n_u32(lo_scaled, 16), vdupq_n_u32(1));
            uint32x4_t hi_lsb = vandq_u32(vshrq_n_u32(hi_scaled, 16), vdupq_n_u32(1));
            lo_scaled = vaddq_u32(lo_scaled, vaddq_u32(vdupq_n_u32(0x7fff), lo_lsb));
            hi_scaled = vaddq_u32(hi_scaled, vaddq_u32(vdupq_n_u32(0x7fff), hi_lsb));
            vst1q_u16(row_ptr + column, vcombine_u16(vshrn_n_u32(lo_scaled, 16), vshrn_n_u32(hi_scaled, 16)));
        }
        for (; column < length; column++) {
            row_ptr[column] = tensor2_f32_to_bf16(tensor2_bf16_to_f32(row_ptr[column]) * factor);
        }
    }
    return TENSOR2_OK;
#elif defined(__AVX2__)
    __m256 scale = _mm256_set1_ps(factor);
    __m256i round_bias = _mm256_set1_epi32(0x7fff);
    __m256i one = _mm256_set1_epi32(1);
    for (int row = 0; row < rows; row++) {
        uint16_t *row_ptr = target + row * stride + offset;
        int column = 0;
        int upper = (length / 8) * 8;
        for (; column < upper; column += 8) {
            __m128i raw16 = _mm_loadu_si128((const __m128i *) (row_ptr + column));
            __m256i bits = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw16), 16);
            __m256 values = _mm256_castsi256_ps(bits);
            __m256 scaled = _mm256_mul_ps(values, scale);
            __m256i scaled_bits = _mm256_castps_si256(scaled);
            __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(scaled_bits, 16), one);
            scaled_bits = _mm256_add_epi32(scaled_bits, _mm256_add_epi32(round_bias, lsb));
            uint32_t shifted[8] __attribute__((aligned(32)));
            _mm256_store_si256((__m256i *) shifted, _mm256_srli_epi32(scaled_bits, 16));
            for (int lane = 0; lane < 8; lane++) {
                row_ptr[column + lane] = (uint16_t) shifted[lane];
            }
        }
        for (; column < length; column++) {
            row_ptr[column] = tensor2_f32_to_bf16(tensor2_bf16_to_f32(row_ptr[column]) * factor);
        }
    }
    return TENSOR2_OK;
#else
    for (int row = 0; row < rows; row++) {
        uint16_t *row_ptr = target + row * stride + offset;
        for (int column = 0; column < length; column++) {
            row_ptr[column] = tensor2_f32_to_bf16(tensor2_bf16_to_f32(row_ptr[column]) * factor);
        }
    }
    return TENSOR2_OK;
#endif
}
