#include "tensor2_native.h"
#include "../simd/vector_simd.h"

#include <string.h>
#include <math.h>

#define TENSOR2_Q8_BLOCK_SIZE 32

static inline float tensor2_bf16_to_f32(uint16_t value);

tensor2_status tensor2_exp_f32(const float *input, float *output, int rows, int offset, int length,
        int input_stride, int output_stride) {
    if (input == NULL || output == NULL || rows < 0 || offset < 0 || length < 0
            || input_stride < 0 || output_stride < 0) {
        return TENSOR2_UNSUPPORTED;
    }
    for (int row = 0; row < rows; row++) {
        const float *input_row = input + row * input_stride;
        float *output_row = output + row * output_stride;
        for (int column = offset; column < offset + length; column++) {
            output_row[column] = expf(input_row[column]);
        }
    }
    return TENSOR2_OK;
}

float tensor2_sum_f32(const float *input, int row, int offset, int length, int input_stride) {
    if (input == NULL || row < 0 || offset < 0 || length <= 0 || input_stride < 0) {
        return 0.0f;
    }
    const float *input_row = input + row * input_stride;
    float sum = 0.0f;
    for (int column = offset; column < offset + length; column++) {
        sum += input_row[column];
    }
    return sum;
}

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

tensor2_status tensor2_batch_dot_f32_bf16(float *result, const float *a, const uint16_t *b,
        int result_rows, int a_row_offset, int a_column_offset, int b_column_offset, int column_length,
        int result_row_offset, int b_row_offset, int row_chunk_size, int result_stride, int a_stride,
        int b_stride) {
    if (result == NULL || a == NULL || b == NULL || result_rows < 0 || a_row_offset < 0
            || a_column_offset < 0 || b_column_offset < 0 || column_length <= 0 || result_row_offset < 0
            || b_row_offset < 0 || row_chunk_size < 0 || result_stride < 0 || a_stride < 0 || b_stride < 0) {
        return TENSOR2_UNSUPPORTED;
    }
#if defined(TENSOR2_ARM_NEON)
    for (int result_row = 0; result_row < result_rows; result_row++) {
        const float *a_row = a + (a_row_offset + result_row) * a_stride + a_column_offset;
        for (int b_row = b_row_offset; b_row < b_row_offset + row_chunk_size; b_row++) {
            const uint16_t *b_row_values = b + b_row * b_stride + b_column_offset;
            float32x4_t sum0 = vdupq_n_f32(0.0f);
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            int column = 0;
            for (; column + 8 <= column_length; column += 8) {
                float32x4_t av0 = vld1q_f32(a_row + column);
                float32x4_t av1 = vld1q_f32(a_row + column + 4);
                uint16x8_t raw = vld1q_u16(b_row_values + column);
                uint32x4_t raw0 = vshlq_n_u32(vmovl_u16(vget_low_u16(raw)), 16);
                uint32x4_t raw1 = vshlq_n_u32(vmovl_u16(vget_high_u16(raw)), 16);
                sum0 = vfmaq_f32(sum0, av0, vreinterpretq_f32_u32(raw0));
                sum1 = vfmaq_f32(sum1, av1, vreinterpretq_f32_u32(raw1));
            }
            float sum = vaddvq_f32(vaddq_f32(sum0, sum1));
            for (; column < column_length; column++) {
                sum += a_row[column] * tensor2_bf16_to_f32(b_row_values[column]);
            }
            result[result_row * result_stride + result_row_offset + b_row] = sum;
        }
    }
    return TENSOR2_OK;
#elif defined(__AVX2__)
    for (int result_row = 0; result_row < result_rows; result_row++) {
        const float *a_row = a + (a_row_offset + result_row) * a_stride + a_column_offset;
        for (int b_row = b_row_offset; b_row < b_row_offset + row_chunk_size; b_row++) {
            const uint16_t *b_row_values = b + b_row * b_stride + b_column_offset;
            __m256 sum = _mm256_setzero_ps();
            int column = 0;
            for (; column + 8 <= column_length; column += 8) {
                __m128i raw = _mm_loadu_si128((const __m128i *) (b_row_values + column));
                __m256i low = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16);
                __m256 values = _mm256_castsi256_ps(low);
                __m256 products = _mm256_mul_ps(_mm256_loadu_ps(a_row + column), values);
                sum = _mm256_add_ps(sum, products);
            }
            float lanes[8] __attribute__((aligned(32)));
            _mm256_store_ps(lanes, sum);
            float scalar = lanes[0] + lanes[1] + lanes[2] + lanes[3]
                    + lanes[4] + lanes[5] + lanes[6] + lanes[7];
            for (; column < column_length; column++) {
                scalar += a_row[column] * tensor2_bf16_to_f32(b_row_values[column]);
            }
            result[result_row * result_stride + result_row_offset + b_row] = scalar;
        }
    }
    return TENSOR2_OK;
#else
    return TENSOR2_UNSUPPORTED;
#endif
}

tensor2_status tensor2_gemm_f32_bf16(float *result, const float *a, const uint16_t *b,
        int result_rows, int a_row_offset, int a_column_offset, int b_column_offset, int column_length,
        int result_row_offset, int b_row_offset, int row_chunk_size, int result_stride, int a_stride,
        int b_stride) {
    if (result == NULL || a == NULL || b == NULL || result_rows < 0 || a_row_offset < 0
            || a_column_offset < 0 || b_column_offset < 0 || column_length <= 0 || result_row_offset < 0
            || b_row_offset < 0 || row_chunk_size < 0 || result_stride < 0 || a_stride < 0 || b_stride < 0) {
        return TENSOR2_UNSUPPORTED;
    }
#if defined(TENSOR2_ARM_NEON)
    for (int result_row = 0; result_row < result_rows; result_row++) {
        const float *a_row = a + (a_row_offset + result_row) * a_stride + a_column_offset;
        int b_row = b_row_offset;
        for (; b_row + 4 <= b_row_offset + row_chunk_size; b_row += 4) {
            const uint16_t *b0 = b + (b_row + 0) * b_stride + b_column_offset;
            const uint16_t *b1 = b + (b_row + 1) * b_stride + b_column_offset;
            const uint16_t *b2 = b + (b_row + 2) * b_stride + b_column_offset;
            const uint16_t *b3 = b + (b_row + 3) * b_stride + b_column_offset;
            float32x4_t lo0 = vdupq_n_f32(0.0f), lo1 = vdupq_n_f32(0.0f);
            float32x4_t lo2 = vdupq_n_f32(0.0f), lo3 = vdupq_n_f32(0.0f);
            int column = 0;
            for (; column + 8 <= column_length; column += 8) {
                float32x4_t av0 = vld1q_f32(a_row + column);
                float32x4_t av1 = vld1q_f32(a_row + column + 4);
                uint16x8_t r0 = vld1q_u16(b0 + column);
                uint16x8_t r1 = vld1q_u16(b1 + column);
                uint16x8_t r2 = vld1q_u16(b2 + column);
                uint16x8_t r3 = vld1q_u16(b3 + column);
                lo0 = vfmaq_f32(lo0, av0, vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(r0)), 16)));
                lo1 = vfmaq_f32(lo1, av0, vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(r1)), 16)));
                lo2 = vfmaq_f32(lo2, av0, vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(r2)), 16)));
                lo3 = vfmaq_f32(lo3, av0, vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(r3)), 16)));
                lo0 = vfmaq_f32(lo0, av1, vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(r0)), 16)));
                lo1 = vfmaq_f32(lo1, av1, vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(r1)), 16)));
                lo2 = vfmaq_f32(lo2, av1, vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(r2)), 16)));
                lo3 = vfmaq_f32(lo3, av1, vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(r3)), 16)));
            }
            float sums[4] = {vaddvq_f32(lo0), vaddvq_f32(lo1), vaddvq_f32(lo2), vaddvq_f32(lo3)};
            for (; column < column_length; column++) {
                float value = a_row[column];
                sums[0] += value * tensor2_bf16_to_f32(b0[column]);
                sums[1] += value * tensor2_bf16_to_f32(b1[column]);
                sums[2] += value * tensor2_bf16_to_f32(b2[column]);
                sums[3] += value * tensor2_bf16_to_f32(b3[column]);
            }
            result[result_row * result_stride + result_row_offset + b_row + 0] = sums[0];
            result[result_row * result_stride + result_row_offset + b_row + 1] = sums[1];
            result[result_row * result_stride + result_row_offset + b_row + 2] = sums[2];
            result[result_row * result_stride + result_row_offset + b_row + 3] = sums[3];
        }
        if (b_row < b_row_offset + row_chunk_size) {
            tensor2_status status = tensor2_batch_dot_f32_bf16(result + result_row * result_stride, a,
                    b, 1, a_row_offset + result_row, a_column_offset, b_column_offset, column_length,
                    result_row_offset, b_row, b_row_offset + row_chunk_size - b_row, result_stride, a_stride,
                    b_stride);
            if (status != TENSOR2_OK) {
                return status;
            }
        }
    }
    return TENSOR2_OK;
#else
    return TENSOR2_UNSUPPORTED;
#endif
}

tensor2_status tensor2_gemm_f32_q4(float *result, const float *a, const uint8_t *b, const float *b_scales,
        int result_rows, int a_row_offset, int a_column_offset, int b_column_offset, int column_length,
        int result_row_offset, int b_row_offset, int row_chunk_size, int result_stride, int a_stride,
        int b_stride, int b_scale_stride) {
    if (result == 0 || a == 0 || b == 0 || b_scales == 0 || result_rows < 0 || a_row_offset < 0
            || a_column_offset < 0 || b_column_offset < 0 || column_length <= 0 || result_row_offset < 0
            || b_row_offset < 0 || row_chunk_size < 0 || result_stride < 0 || a_stride < 0
            || b_stride < 0 || b_scale_stride < 0 || a_column_offset % 32 != 0
            || b_column_offset % 32 != 0 || column_length % 32 != 0) {
        return TENSOR2_UNSUPPORTED;
    }
    gemm_f32_q4(0, a, a_row_offset * a_stride + a_column_offset, b_scales,
            (const char *) b, b_column_offset / 2, result, -result_row_offset, result_rows,
            b_row_offset, row_chunk_size, column_length, a_stride, b_stride / 2,
            b_scale_stride, result_stride);
    return TENSOR2_OK;
}

tensor2_status tensor2_gemm_i8_q4(float *result, const int8_t *a, const float *a_scales, const uint8_t *b,
        const float *b_scales, int result_rows, int a_row_offset, int a_column_offset, int b_column_offset,
        int column_length, int result_row_offset, int b_row_offset, int row_chunk_size, int result_stride,
        int a_stride, int a_scale_stride, int b_stride, int b_scale_stride) {
    if (result == 0 || a == 0 || a_scales == 0 || b == 0 || b_scales == 0 || result_rows < 0
            || a_row_offset < 0 || a_column_offset < 0 || b_column_offset < 0 || column_length <= 0
            || result_row_offset < 0 || b_row_offset < 0 || row_chunk_size < 0 || result_stride < 0
            || a_stride < 0 || a_scale_stride < 0 || b_stride < 0 || b_scale_stride < 0
            || a_column_offset % 32 != 0 || b_column_offset % 32 != 0 || column_length % 32 != 0) {
        return TENSOR2_UNSUPPORTED;
    }
    gemm_q8_q4(0, a_scales, (const char *) a, a_row_offset * a_stride + a_column_offset,
            b_scales, (const char *) b, b_column_offset / 2, result, -result_row_offset,
            result_rows, b_row_offset, row_chunk_size, column_length, a_stride, a_scale_stride,
            b_stride / 2, b_scale_stride, result_stride);
    return TENSOR2_OK;
}

tensor2_status tensor2_dot_product_batch_chunk_i8_q4(
        float *result0, float *result1, const int8_t *input, const float *input_scales,
        const uint8_t *weights0, const float *weight_scales0, const uint8_t *weights1,
        const float *weight_scales1, int result_rows, int input_column_start, int weight_column_start,
        int column_length, int weight_row_start, int weight_row_count, int result_column_start,
        int result0_stride, int result1_stride, int input_stride, int input_scale_stride,
        int weight0_stride, int weight0_scale_stride, int weight1_stride, int weight1_scale_stride) {
    if (result0 == NULL || result1 == NULL || input == NULL || input_scales == NULL
            || weights0 == NULL || weight_scales0 == NULL || weights1 == NULL || weight_scales1 == NULL) {
        return TENSOR2_UNSUPPORTED;
    }
    tensor2_status first = tensor2_dot_product_rows_i8_q4(result0, input, input_scales, weights0, weight_scales0,
            result_rows, input_column_start, weight_column_start, column_length, weight_row_start,
            weight_row_count, result_column_start, result0_stride, input_stride, input_scale_stride,
            weight0_stride, weight0_scale_stride);
    if (first != TENSOR2_OK) {
        return first;
    }
    return tensor2_dot_product_rows_i8_q4(result1, input, input_scales, weights1, weight_scales1,
            result_rows, input_column_start, weight_column_start, column_length, weight_row_start,
            weight_row_count, result_column_start, result1_stride, input_stride, input_scale_stride,
            weight1_stride, weight1_scale_stride);
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

tensor2_status tensor2_dot_product_rows_f32_f32(
        float *result,
        const float *input,
        const float *weights,
        int result_rows,
        int input_start,
        int input_length,
        int weight_row_start,
        int weight_row_count,
        int result_column_start,
        int result_stride,
        int input_stride,
        int weight_stride) {
    if (result == 0 || input == 0 || weights == 0) {
        return TENSOR2_UNSUPPORTED;
    }
    if (result_rows < 0 || input_start < 0 || input_length < 0 || weight_row_start < 0
            || weight_row_count < 0 || result_column_start < 0 || result_stride < 0
            || input_stride < 0 || weight_stride < 0) {
        return TENSOR2_UNSUPPORTED;
    }

    for (int result_row = 0; result_row < result_rows; result_row++) {
        const int input_offset = result_row * input_stride + input_start;
        for (int row = 0; row < weight_row_count; row++) {
            const int weight_row = weight_row_start + row;
            const int weight_offset = weight_row * weight_stride + input_start;
            result[result_row * result_stride + result_column_start + row] =
                    tensor2_dot_f32_f32(input, weights, input_offset, weight_offset, input_length);
        }
    }
    return TENSOR2_OK;
}

tensor2_status tensor2_dot_product_rows_f32_q8(
        float *result,
        const float *input,
        const int8_t *weights,
        const float *weight_scales,
        int result_rows,
        int input_start,
        int input_length,
        int weight_row_start,
        int weight_row_count,
        int result_column_start,
        int result_stride,
        int input_stride,
        int weight_stride,
        int weight_scale_stride) {
    if (result == 0 || input == 0 || weights == 0 || weight_scales == 0) {
        return TENSOR2_UNSUPPORTED;
    }
    if (result_rows < 0 || input_start < 0 || input_length < 0 || weight_row_start < 0
            || weight_row_count < 0 || result_column_start < 0 || result_stride < 0
            || input_stride < 0 || weight_stride < 0 || weight_scale_stride < 0
            || input_start % TENSOR2_Q8_BLOCK_SIZE != 0
            || input_length % TENSOR2_Q8_BLOCK_SIZE != 0) {
        return TENSOR2_UNSUPPORTED;
    }

#if !defined(TENSOR2_ARM_NEON) && !defined(__AVX2__)
    return TENSOR2_UNSUPPORTED;
#else
    int scale_offset = input_start / TENSOR2_Q8_BLOCK_SIZE;
    for (int result_row = 0; result_row < result_rows; result_row++) {
        const float *input_row = input + result_row * input_stride + input_start;
        for (int row = 0; row < weight_row_count; row++) {
            const int weight_row = weight_row_start + row;
            const int8_t *weight_row_ptr = weights + weight_row * weight_stride + input_start;
            const float *scale_row = weight_scales + weight_row * weight_scale_stride + scale_offset;
            result[result_row * result_stride + result_column_start + row] =
                    tensor2_dot_f32_q8(input_row, weight_row_ptr, scale_row, input_length);
        }
    }
    return TENSOR2_OK;
#endif
}

static inline float tensor2_dot_bf16_q4(const uint16_t *input, const uint8_t *weights,
        const float *scales, int column_length) {
#if defined(TENSOR2_ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int8_t low[16] __attribute__((aligned(16)));
    int8_t high[16] __attribute__((aligned(16)));
    for (int block = 0; block < column_length / 32; block++) {
        uint8x16_t packed = vld1q_u8(weights + block * 16);
        vst1q_s8(low, vreinterpretq_s8_u8(vsubq_u8(vandq_u8(packed, vdupq_n_u8(15)), vdupq_n_u8(8))));
        vst1q_s8(high, vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(packed, 4), vdupq_n_u8(8))));
        float32x4_t scale = vdupq_n_f32(scales[block]);
        const uint16_t *values = input + block * 32;
        for (int group = 0; group < 4; group++) {
            int base = group * 4;
            int32x4_t lowQ = vmovl_s16(vget_low_s16(vmovl_s8(vld1_s8(low + base))));
            int32x4_t highQ = vmovl_s16(vget_low_s16(vmovl_s8(vld1_s8(high + base))));
            uint32x4_t lowBits = vshlq_n_u32(vmovl_u16(vld1_u16(values + base)), 16);
            uint32x4_t highBits = vshlq_n_u32(vmovl_u16(vld1_u16(values + base + 16)), 16);
            acc = vmlaq_f32(acc, vreinterpretq_f32_u32(lowBits), vmulq_f32(vcvtq_f32_s32(lowQ), scale));
            acc = vmlaq_f32(acc, vreinterpretq_f32_u32(highBits), vmulq_f32(vcvtq_f32_s32(highQ), scale));
        }
    }
    return tensor2_hsum4(acc);
#else
    float sum = 0.0f;
    for (int block = 0; block < column_length / 32; block++) {
        float scale = scales[block];
        const uint8_t *packed = weights + block * 16;
        const uint16_t *values = input + block * 32;
        for (int column = 0; column < 16; column++) {
            uint8_t value = packed[column];
            sum += tensor2_bf16_to_f32(values[column]) * ((float) ((value & 15) - 8)) * scale;
            sum += tensor2_bf16_to_f32(values[column + 16]) * ((float) ((value >> 4) - 8)) * scale;
        }
    }
    return sum;
#endif
}

tensor2_status tensor2_dot_product_rows_bf16_q4(
        void *result,
        const uint16_t *input,
        const uint8_t *weights,
        const float *weight_scales,
        int result_bf16,
        int result_rows,
        int input_column_start,
        int weight_column_start,
        int column_length,
        int weight_row_start,
        int weight_row_count,
        int result_column_start,
        int result_stride,
        int input_stride,
        int weight_stride,
        int weight_scale_stride) {
    if (result == 0 || input == 0 || weights == 0 || weight_scales == 0
            || input_column_start % 32 != 0 || weight_column_start % 32 != 0 || column_length % 32 != 0) {
        return TENSOR2_UNSUPPORTED;
    }
    int weightBlockOffset = weight_column_start / 32;
    for (int resultRow = 0; resultRow < result_rows; resultRow++) {
        const uint16_t *inputRow = input + resultRow * input_stride + input_column_start;
        for (int row = 0; row < weight_row_count; row++) {
            int weightRow = weight_row_start + row;
            const uint8_t *packed = weights + weightRow * weight_stride / 2 + weightBlockOffset * 16;
            const float *scales = weight_scales + weightRow * weight_scale_stride + weightBlockOffset;
            float sum = tensor2_dot_bf16_q4(inputRow, packed, scales, column_length);
            if (result_bf16) {
                ((uint16_t *) result)[resultRow * result_stride + result_column_start + row]
                        = tensor2_f32_to_bf16(sum);
            } else {
                ((float *) result)[resultRow * result_stride + result_column_start + row] = sum;
            }
        }
    }
    return TENSOR2_OK;
}

static inline float tensor2_dot_f32_q4(const float *input, const uint8_t *weights,
        const float *scales, int column_length) {
#if defined(TENSOR2_ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int8_t low[16] __attribute__((aligned(16)));
    int8_t high[16] __attribute__((aligned(16)));
    for (int block = 0; block < column_length / 32; block++) {
        uint8x16_t packed = vld1q_u8(weights + block * 16);
        vst1q_s8(low, vreinterpretq_s8_u8(vsubq_u8(vandq_u8(packed, vdupq_n_u8(15)), vdupq_n_u8(8))));
        vst1q_s8(high, vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(packed, 4), vdupq_n_u8(8))));
        float32x4_t scale = vdupq_n_f32(scales[block]);
        const float *values = input + block * 32;
        for (int group = 0; group < 4; group++) {
            int base = group * 4;
            int32x4_t lowQ = vmovl_s16(vget_low_s16(vmovl_s8(vld1_s8(low + base))));
            int32x4_t highQ = vmovl_s16(vget_low_s16(vmovl_s8(vld1_s8(high + base))));
            acc = vmlaq_f32(acc, vld1q_f32(values + base), vmulq_f32(vcvtq_f32_s32(lowQ), scale));
            acc = vmlaq_f32(acc, vld1q_f32(values + base + 16), vmulq_f32(vcvtq_f32_s32(highQ), scale));
        }
    }
    return tensor2_hsum4(acc);
#else
    float sum = 0.0f;
    int blocks = column_length / 32;
    for (int block = 0; block < blocks; block++) {
        float scale = scales[block];
        const uint8_t *packed = weights + block * 16;
        const float *values = input + block * 32;
        for (int column = 0; column < 16; column++) {
            uint8_t value = packed[column];
            sum += values[column] * ((float) ((value & 0x0f) - 8)) * scale;
            sum += values[column + 16] * ((float) ((value >> 4) - 8)) * scale;
        }
    }
    return sum;
#endif
}

tensor2_status tensor2_dot_product_rows_f32_q4(
        float *result,
        const float *input,
        const uint8_t *weights,
        const float *weight_scales,
        int result_rows,
        int input_column_start,
        int weight_column_start,
        int column_length,
        int weight_row_start,
        int weight_row_count,
        int result_column_start,
        int result_stride,
        int input_stride,
        int weight_stride,
        int weight_scale_stride) {
    if (result == 0 || input == 0 || weights == 0 || weight_scales == 0) {
        return TENSOR2_UNSUPPORTED;
    }
    if (result_rows < 0 || input_column_start < 0 || weight_column_start < 0 || column_length < 0
            || weight_row_start < 0 || weight_row_count < 0 || result_column_start < 0
            || result_stride < 0 || input_stride < 0 || weight_stride < 0 || weight_scale_stride < 0
            || input_column_start % 32 != 0 || weight_column_start % 32 != 0 || column_length % 32 != 0) {
        return TENSOR2_UNSUPPORTED;
    }
    int weightBlockOffset = weight_column_start / 32;
    for (int result_row = 0; result_row < result_rows; result_row++) {
        const float *inputRow = input + result_row * input_stride + input_column_start;
        for (int row = 0; row < weight_row_count; row++) {
            int weightRow = weight_row_start + row;
            const uint8_t *weightRowPtr = weights + weightRow * weight_stride / 2 + weightBlockOffset * 16;
            const float *scaleRow = weight_scales + weightRow * weight_scale_stride + weightBlockOffset;
            result[result_row * result_stride + result_column_start + row] =
                    tensor2_dot_f32_q4(inputRow, weightRowPtr, scaleRow, column_length);
        }
    }
    return TENSOR2_OK;
}

tensor2_status tensor2_dot_product_rows_i8_q4(
        float *result,
        const int8_t *input,
        const float *input_scales,
        const uint8_t *weights,
        const float *weight_scales,
        int result_rows,
        int input_column_start,
        int weight_column_start,
        int column_length,
        int weight_row_start,
        int weight_row_count,
        int result_column_start,
        int result_stride,
        int input_stride,
        int input_scale_stride,
        int weight_stride,
        int weight_scale_stride) {
    if (result == 0 || input == 0 || input_scales == 0 || weights == 0 || weight_scales == 0) {
        return TENSOR2_UNSUPPORTED;
    }
    if (result_rows < 0 || input_column_start < 0 || weight_column_start < 0 || column_length < 0
            || weight_row_start < 0 || weight_row_count < 0 || result_column_start < 0
            || result_stride < 0 || input_stride < 0 || input_scale_stride < 0
            || weight_stride < 0 || weight_scale_stride < 0
            || input_column_start % Q8_BLOCK_SIZE != 0
            || weight_column_start % Q4_BLOCK_SIZE != 0
            || column_length % Q8_BLOCK_SIZE != 0
            || weight_stride % 2 != 0) {
        return TENSOR2_UNSUPPORTED;
    }
    if (result_rows == 0 || weight_row_count == 0) {
        return TENSOR2_OK;
    }
    gemm_q8_q4(0, input_scales, (const char *) input, input_column_start,
            weight_scales, (const char *) weights, weight_column_start / 2,
            result, weight_row_start - result_column_start, result_rows, weight_row_start,
            weight_row_count, column_length, input_stride, input_scale_stride, weight_stride / 2,
            weight_scale_stride, result_stride);
    return TENSOR2_OK;
}

tensor2_status tensor2_saxpy_f32(
        float alpha,
        const float *x,
        float *y,
        int x_offset,
        int y_offset,
        int length) {
    if (x == 0 || y == 0 || x_offset < 0 || y_offset < 0 || length < 0) {
        return TENSOR2_UNSUPPORTED;
    }
#if defined(TENSOR2_ARM_NEON)
    float32x4_t factor = vdupq_n_f32(alpha);
    int i = 0;
    int upper = (length / 4) * 4;
    for (; i < upper; i += 4) {
        float32x4_t xv = vld1q_f32(x + x_offset + i);
        float32x4_t yv = vld1q_f32(y + y_offset + i);
        vst1q_f32(y + y_offset + i, vmlaq_f32(yv, xv, factor));
    }
#elif defined(__AVX2__)
    __m256 factor = _mm256_set1_ps(alpha);
    int i = 0;
    int upper = (length / 8) * 8;
    for (; i < upper; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + x_offset + i);
        __m256 yv = _mm256_loadu_ps(y + y_offset + i);
        _mm256_storeu_ps(y + y_offset + i, _mm256_add_ps(yv, _mm256_mul_ps(xv, factor)));
    }
#else
    int i = 0;
#endif
    for (; i < length; i++) {
        y[y_offset + i] += alpha * x[x_offset + i];
    }
    return TENSOR2_OK;
}

tensor2_status tensor2_saxpy_f32_batch(
        const float *alpha,
        const float *x,
        float *y,
        int x_offset,
        int y_offset,
        int length,
        int alpha_offset,
        int x_row_offset,
        int batch_size,
        int x_stride) {
    if (alpha == 0 || x == 0 || y == 0 || x_offset < 0 || y_offset < 0 || length < 0
            || alpha_offset < 0 || x_row_offset < 0 || batch_size < 0 || x_stride < 0) {
        return TENSOR2_UNSUPPORTED;
    }
    for (int row = 0; row < batch_size; row++) {
        tensor2_status status = tensor2_saxpy_f32(alpha[alpha_offset + row],
                x + (x_row_offset + row) * x_stride, y, x_offset, y_offset, length);
        if (status != TENSOR2_OK) {
            return status;
        }
    }
    return TENSOR2_OK;
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
