/**
 * @file vector_simd.c
 * @brief SIMD accelerated matrix multiplication
 *
 * SIMD accelerated matrix multiplication.  Derived from the work of
 *  J. Tunney, ‘LLaMA Now Goes Faster on CPUs’, Mar. 2024. [Online].
 *  Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].
 */
#include <stdio.h>
#include <stddef.h>
#if defined(__ARM_NEON__)
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif
#include "vector_simd.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

static inline short fp32_to_bf16(float s) {
    uint16_t bf;
    union {
        float f;
        uint32_t i;
    } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        bf = (u.i >> 16) | 64; /* force to quiet */
        return bf;
    }
    if (!(u.i & 0x7f800000)) { /* subnormal */
        bf = (u.i & 0x80000000) >> 16; /* flush to zero */
        return bf;
    }
    bf = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    return bf;
}

static inline float bf16_to_fp32(short s) {
    union {
        uint32_t i;
        float f;
    } u;
    u.i = ((uint32_t) (uint16_t) s) << 16;
    return u.f;
}

#if defined(__ARM_NEON__)
static inline float32x4_t load_bf16x4_as_f32(const short *p) {
    uint16x4_t bf16 = vld1_u16((const uint16_t *) p);
    uint32x4_t f32_bits = vshll_n_u16(bf16, 16);
    return vreinterpretq_f32_u32(f32_bits);
}
#else
static inline __m256 load_bf16x8_as_f32(const short *p) {
    __m128i bf16 = _mm_loadu_si128((const __m128i *) p);
    __m256i f32_bits = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16), 16);
    return _mm256_castsi256_ps(f32_bits);
}

#if defined(__AVX512F__)
static inline __m512 load_bf16x16_as_f32(const short *p) {
    __m256i bf16 = _mm256_loadu_si256((const __m256i *) p);
    __m512i f32_bits = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16), 16);
    return _mm512_castsi512_ps(f32_bits);
}
#endif
#endif

//All params
struct gemm_params {
    int flags;
    const float* restrict af;
    const char* restrict a;
    const short* restrict as;
    int aoffset;
    const float* restrict bf;
    const char* restrict b;
    const short* restrict bs;
    int boffset;
    float * restrict r;
    short * restrict rs;
    int roffset;
    int m;
    int n;
    int k;
    int lda;
    int ldaf;
    int ldb;
    int ldbf;
    int ldc;
} gemm_params;


static void saxpy_f32_scalar(float alpha, const float *x, float *y, int xoffset, int yoffset, int limit) {
    for (int i = 0; i < limit; i++) {
        y[yoffset + i] += alpha * x[xoffset + i];
    }
}

#if defined(__ARM_NEON__)
static void saxpy_f32_128_arm(float alpha, const float *x, float *y, int xoffset, int yoffset, int limit) {
    float32x4_t alpha_vec = vdupq_n_f32(alpha);
    int i = 0;
    for ( ; i + 4 <= limit; i += 4) {
        //load 4 float32 values from memory into one 128-bit NEON vector
        float32x4_t acc = vld1q_f32(y + yoffset + i);
        //load 4 float32 values from memory into one 128-bit NEON vector
        float32x4_t xv = vld1q_f32(x + xoffset + i);
        //is ARM NEON multiply-accumulate for 4 float32 values.
        float32x4_t yv = vmlaq_f32(acc, xv, alpha_vec);
        //store 4 float32 values from yv into memory at y[yoffset + i ... yoffset + i + 3]
        vst1q_f32(y + yoffset + i, yv);
    }
    for (; i < limit; i++) {
        y[yoffset + i] += alpha * x[xoffset + i];
    }
}
#endif

// public void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit)
void saxpy_f32(float alpha, const float *x, float *y, int xoffset, int yoffset, int limit) {
#if defined(__ARM_NEON__)
    saxpy_f32_128_arm(alpha, x, y, xoffset, yoffset, limit);
#else
    saxpy_f32_scalar(alpha, x, y, xoffset, yoffset, limit);
#endif
}

static void saxpy_f32_batch_scalar(const float *alpha, const float *x, float *y, int xoffset, int yoffset, int limit,
                                   int aoffset, int xrowoffset, int batch_size, int xstride) {
    for (int row = 0; row < batch_size; row++) {
        saxpy_f32_scalar(alpha[aoffset + row], x, y, ((xrowoffset + row) * xstride) + xoffset, yoffset, limit);
    }
}

#if defined(__ARM_NEON__)
static void saxpy_f32_batch_128_arm(const float *alpha, const float *x, float *y, int xoffset, int yoffset, int limit,
                                    int aoffset, int xrowoffset, int batch_size, int xstride) {
    int row = 0;
    for (; row + 4 <= batch_size; row += 4) {
        float32x4_t a0 = vdupq_n_f32(alpha[aoffset + row]);
        float32x4_t a1 = vdupq_n_f32(alpha[aoffset + row + 1]);
        float32x4_t a2 = vdupq_n_f32(alpha[aoffset + row + 2]);
        float32x4_t a3 = vdupq_n_f32(alpha[aoffset + row + 3]);
        const float *x0p = x + ((xrowoffset + row) * xstride) + xoffset;
        const float *x1p = x + ((xrowoffset + row + 1) * xstride) + xoffset;
        const float *x2p = x + ((xrowoffset + row + 2) * xstride) + xoffset;
        const float *x3p = x + ((xrowoffset + row + 3) * xstride) + xoffset;
        int i = 0;
        for (; i + 4 <= limit; i += 4) {
            float32x4_t acc = vld1q_f32(y + yoffset + i);
            acc = vmlaq_f32(acc, vld1q_f32(x0p + i), a0);
            acc = vmlaq_f32(acc, vld1q_f32(x1p + i), a1);
            acc = vmlaq_f32(acc, vld1q_f32(x2p + i), a2);
            acc = vmlaq_f32(acc, vld1q_f32(x3p + i), a3);
            vst1q_f32(y + yoffset + i, acc);
        }
        for (; i < limit; i++) {
            y[yoffset + i] += alpha[aoffset + row] * x0p[i]
                    + alpha[aoffset + row + 1] * x1p[i]
                    + alpha[aoffset + row + 2] * x2p[i]
                    + alpha[aoffset + row + 3] * x3p[i];
        }
    }
    for (; row < batch_size; row++) {
        saxpy_f32_128_arm(alpha[aoffset + row], x, y, ((xrowoffset + row) * xstride) + xoffset, yoffset, limit);
    }
}
#endif

void saxpy_f32_batch(const float *alpha, const float *x, float *y, int xoffset, int yoffset, int limit,
                      int aoffset, int xrowoffset, int batch_size, int xstride) {
#if defined(__ARM_NEON__)
    saxpy_f32_batch_128_arm(alpha, x, y, xoffset, yoffset, limit, aoffset, xrowoffset, batch_size, xstride);
#else
    saxpy_f32_batch_scalar(alpha, x, y, xoffset, yoffset, limit, aoffset, xrowoffset, batch_size, xstride);
#endif
}

void saxpy_q8_f32(float alpha, const float *xf, const char *x, float *y, int xoffset, int yoffset, int limit,
                  int xstride, int xscale_stride) {
    int block_start = xoffset / Q8_BLOCK_SIZE;
    int blocks = limit / Q8_BLOCK_SIZE;
#if defined(__ARM_NEON__)
    float32x4_t av = vdupq_n_f32(alpha);
    for (int block = 0; block < blocks; block++) {
        float32x4_t sv = vdupq_n_f32(xf[block_start + block]);
        int base = xoffset + block * Q8_BLOCK_SIZE;
        int ybase = yoffset + block * Q8_BLOCK_SIZE;
        for (int j = 0; j < Q8_BLOCK_SIZE; j += 16) {
            int8x16_t q = vld1q_s8((const int8_t *) (x + base + j));
            int16x8_t qlo = vmovl_s8(vget_low_s8(q));
            int16x8_t qhi = vmovl_s8(vget_high_s8(q));
            float32x4_t q0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qlo))), sv);
            float32x4_t q1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qlo))), sv);
            float32x4_t q2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qhi))), sv);
            float32x4_t q3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qhi))), sv);
            vst1q_f32(y + ybase + j, vmlaq_f32(vld1q_f32(y + ybase + j), q0, av));
            vst1q_f32(y + ybase + j + 4, vmlaq_f32(vld1q_f32(y + ybase + j + 4), q1, av));
            vst1q_f32(y + ybase + j + 8, vmlaq_f32(vld1q_f32(y + ybase + j + 8), q2, av));
            vst1q_f32(y + ybase + j + 12, vmlaq_f32(vld1q_f32(y + ybase + j + 12), q3, av));
        }
    }
#else
    for (int block = 0; block < blocks; block++) {
        float scale = xf[block_start + block];
        int base = xoffset + block * Q8_BLOCK_SIZE;
        int ybase = yoffset + block * Q8_BLOCK_SIZE;
        for (int j = 0; j < Q8_BLOCK_SIZE; j++) {
            y[ybase + j] += alpha * ((float) x[base + j] * scale);
        }
    }
#endif
}

void saxpy_q8_f32_batch(const float *alpha, const float *xf, const char *x, float *y, int xoffset, int yoffset,
                         int limit, int aoffset, int xrowoffset, int batch_size, int xstride, int xscale_stride) {
    for (int row = 0; row < batch_size; row++) {
        int xrow = xrowoffset + row;
        saxpy_q8_f32(alpha[aoffset + row], xf + xrow * xscale_stride, x + xrow * xstride, y,
                     xoffset, yoffset, limit, xstride, xscale_stride);
    }
}

void gemm_f32_q8(int flags, const float *a, int aoffset, const float *bf, const char* b, int boffset,
                 float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc) {
    int blocks = k / Q8_BLOCK_SIZE;
    int bblock_start = boffset / Q8_BLOCK_SIZE;
    for (int row = 0; row < m; row++) {
        for (int out_col = 0; out_col < n; out_col++) {
            int weight_row = n0 + out_col;
#if defined(__ARM_NEON__)
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int block = 0; block < blocks; block++) {
                float32x4_t sv = vdupq_n_f32(bf[weight_row * ldbf + bblock_start + block]);
                int abase = row * lda + aoffset + block * Q8_BLOCK_SIZE;
                int bbase = weight_row * ldb + boffset + block * Q8_BLOCK_SIZE;
                for (int j = 0; j < Q8_BLOCK_SIZE; j += 16) {
                    int8x16_t q = vld1q_s8((const int8_t *) (b + bbase + j));
                    int16x8_t qlo = vmovl_s8(vget_low_s8(q));
                    int16x8_t qhi = vmovl_s8(vget_high_s8(q));
                    float32x4_t q0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qlo))), sv);
                    float32x4_t q1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qlo))), sv);
                    float32x4_t q2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qhi))), sv);
                    float32x4_t q3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qhi))), sv);
                    acc = vmlaq_f32(acc, vld1q_f32(a + abase + j), q0);
                    acc = vmlaq_f32(acc, vld1q_f32(a + abase + j + 4), q1);
                    acc = vmlaq_f32(acc, vld1q_f32(a + abase + j + 8), q2);
                    acc = vmlaq_f32(acc, vld1q_f32(a + abase + j + 12), q3);
                }
            }
            float sum = vaddvq_f32(acc);
#else
            float sum = 0.0f;
            for (int block = 0; block < blocks; block++) {
                float scale = bf[weight_row * ldbf + bblock_start + block];
                int abase = row * lda + aoffset + block * Q8_BLOCK_SIZE;
                int bbase = weight_row * ldb + boffset + block * Q8_BLOCK_SIZE;
                for (int j = 0; j < Q8_BLOCK_SIZE; j++) {
                    sum += a[abase + j] * (float) b[bbase + j] * scale;
                }
            }
#endif
            r[row * ldc + weight_row - roffset] = sum;
        }
    }
}

void exp_f32(const float *input, float *output, int rows, int offset, int length, int input_stride, int output_stride) {
    for (int row = 0; row < rows; row++) {
        const float *in = input + row * input_stride + offset;
        float *out = output + row * output_stride + offset;
#if defined(__APPLE__)
        int n = length;
        vvexpf(out, in, &n);
#else
        for (int i = 0; i < length; i++) {
            out[i] = expf(in[i]);
        }
#endif
    }
}

float max_f32(const float *input, int row, int offset, int length, int input_stride) {
    const float *in = input + row * input_stride + offset;
    int i = 0;
#if defined(__ARM_NEON__)
    float32x4_t maxv = vdupq_n_f32(-INFINITY);
    for (; i + 4 <= length; i += 4) {
        maxv = vmaxq_f32(maxv, vld1q_f32(in + i));
    }
    float max = vmaxvq_f32(maxv);
#elif defined(__AVX512F__)
    __m512 maxv = _mm512_set1_ps(-INFINITY);
    for (; i + 16 <= length; i += 16) {
        maxv = _mm512_max_ps(maxv, _mm512_loadu_ps(in + i));
    }
    float max = _mm512_reduce_max_ps(maxv);
#else
    __m256 maxv = _mm256_set1_ps(-INFINITY);
    for (; i + 8 <= length; i += 8) {
        maxv = _mm256_max_ps(maxv, _mm256_loadu_ps(in + i));
    }
    __attribute__((aligned(32))) float tmp[8];
    _mm256_store_ps(tmp, maxv);
    float max = tmp[0];
    for (int j = 1; j < 8; j++) {
        if (tmp[j] > max) {
            max = tmp[j];
        }
    }
#endif
    for (; i < length; i++) {
        if (in[i] > max) {
            max = in[i];
        }
    }
    return max;
}

float sum_f32(const float *input, int row, int offset, int length, int input_stride) {
    const float *in = input + row * input_stride + offset;
    int i = 0;
#if defined(__ARM_NEON__)
    float32x4_t sumv = vdupq_n_f32(0.0f);
    for (; i + 4 <= length; i += 4) {
        sumv = vaddq_f32(sumv, vld1q_f32(in + i));
    }
    float sum = vaddvq_f32(sumv);
#elif defined(__AVX512F__)
    __m512 sumv = _mm512_set1_ps(0.0f);
    for (; i + 16 <= length; i += 16) {
        sumv = _mm512_add_ps(sumv, _mm512_loadu_ps(in + i));
    }
    float sum = _mm512_reduce_add_ps(sumv);
#else
    __m256 sumv = _mm256_set1_ps(0.0f);
    for (; i + 8 <= length; i += 8) {
        sumv = _mm256_add_ps(sumv, _mm256_loadu_ps(in + i));
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, sumv);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
#endif
    for (; i < length; i++) {
        sum += in[i];
    }
    return sum;
}

void argmax_f32(const float *input, float *output, int row, int offset, int length, int input_stride) {
    const float *base = input + row * input_stride;
    const int limit = offset + length;
    int max_index = offset;
    float max_value = base[offset];
    int i = offset;
#if defined(__ARM_NEON__)
    if (length >= 4) {
        const uint32_t lane_init[4] = {0, 1, 2, 3};
        const uint32x4_t lanes = vld1q_u32(lane_init);
        float32x4_t maxv = vld1q_f32(base + offset);
        uint32x4_t idxv = vaddq_u32(vdupq_n_u32((uint32_t) offset), lanes);
        i = offset + 4;
        for (; i + 4 <= limit; i += 4) {
            float32x4_t values = vld1q_f32(base + i);
            uint32x4_t indices = vaddq_u32(vdupq_n_u32((uint32_t) i), lanes);
            uint32x4_t greater = vcgtq_f32(values, maxv);
            maxv = vbslq_f32(greater, values, maxv);
            idxv = vbslq_u32(greater, indices, idxv);
        }
        float values[4];
        uint32_t indices[4];
        vst1q_f32(values, maxv);
        vst1q_u32(indices, idxv);
        max_value = values[0];
        max_index = (int) indices[0];
        for (int lane = 1; lane < 4; lane++) {
            if (values[lane] > max_value || (values[lane] == max_value && (int) indices[lane] < max_index)) {
                max_value = values[lane];
                max_index = (int) indices[lane];
            }
        }
    } else {
        i = offset + 1;
    }
#elif defined(__AVX512F__)
    if (length >= 16) {
        const __m512i lanes = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        __m512 maxv = _mm512_loadu_ps(base + offset);
        __m512i idxv = _mm512_add_epi32(_mm512_set1_epi32(offset), lanes);
        i = offset + 16;
        for (; i + 16 <= limit; i += 16) {
            __m512 values = _mm512_loadu_ps(base + i);
            __m512i indices = _mm512_add_epi32(_mm512_set1_epi32(i), lanes);
            __mmask16 greater = _mm512_cmp_ps_mask(values, maxv, _CMP_GT_OQ);
            maxv = _mm512_mask_blend_ps(greater, maxv, values);
            idxv = _mm512_mask_blend_epi32(greater, idxv, indices);
        }
        float values[16];
        int indices[16];
        _mm512_storeu_ps(values, maxv);
        _mm512_storeu_si512((__m512i *) indices, idxv);
        max_value = values[0];
        max_index = indices[0];
        for (int lane = 1; lane < 16; lane++) {
            if (values[lane] > max_value || (values[lane] == max_value && indices[lane] < max_index)) {
                max_value = values[lane];
                max_index = indices[lane];
            }
        }
    } else {
        i = offset + 1;
    }
#else
    if (length >= 8) {
        const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256 maxv = _mm256_loadu_ps(base + offset);
        __m256i idxv = _mm256_add_epi32(_mm256_set1_epi32(offset), lanes);
        i = offset + 8;
        for (; i + 8 <= limit; i += 8) {
            __m256 values = _mm256_loadu_ps(base + i);
            __m256i indices = _mm256_add_epi32(_mm256_set1_epi32(i), lanes);
            __m256 greater = _mm256_cmp_ps(values, maxv, _CMP_GT_OQ);
            maxv = _mm256_blendv_ps(maxv, values, greater);
            idxv = _mm256_blendv_epi8(idxv, indices, _mm256_castps_si256(greater));
        }
        float values[8];
        int indices[8];
        _mm256_storeu_ps(values, maxv);
        _mm256_storeu_si256((__m256i *) indices, idxv);
        max_value = values[0];
        max_index = indices[0];
        for (int lane = 1; lane < 8; lane++) {
            if (values[lane] > max_value || (values[lane] == max_value && indices[lane] < max_index)) {
                max_value = values[lane];
                max_index = indices[lane];
            }
        }
    } else {
        i = offset + 1;
    }
#endif
    for (; i < limit; i++) {
        float value = base[i];
        if (value > max_value) {
            max_value = value;
            max_index = i;
        }
    }
    output[0] = (float) max_index;
    output[1] = max_value;
}

void activation_multiply_quantize_silu_q8(const float *gate, const float *up, char *out, float *out_scale,
    int rows, int offset, int length, int gate_stride, int up_stride, int out_stride, int scale_stride) {
    float block[Q8_BLOCK_SIZE];
    for (int row = 0; row < rows; row++) {
        const float *gate_row = gate + row * gate_stride;
        const float *up_row = up + row * up_stride;
        char *out_row = out + row * out_stride;
        float *scale_row = out_scale + row * scale_stride;
        for (int col = offset; col < offset + length; col += Q8_BLOCK_SIZE) {
            float max_abs = 0.0f;
            for (int i = 0; i < Q8_BLOCK_SIZE; i++) {
                float g = gate_row[col + i];
                float silu = g * (1.0f / (1.0f + expf(-g)));
                float v = silu * up_row[col + i];
                block[i] = v;
                float av = fabsf(v);
                if (av > max_abs) {
                    max_abs = av;
                }
            }
            float scale = max_abs / 127.0f;
            float inv_scale = max_abs != 0.0f ? 127.0f / max_abs : 0.0f;
            scale_row[col / Q8_BLOCK_SIZE] = scale;
            for (int i = 0; i < Q8_BLOCK_SIZE; i++) {
                out_row[col + i] = (char) (block[i] * inv_scale + 0.5f);
            }
        }
    }
}

void __attribute__((noinline)) gemm(int m0, int m, int n0, int n,
  void (*gemmPtr)(int, int, int, int, int, int, struct gemm_params),
  struct gemm_params params) {
    int mc, nc, mp, np;
    switch ((MIN(m - m0, 5) << 4) | MIN(n - n0, 5)) {
            case 0x55:
                mc = 5;
                nc = 5;
                break;
            case 0x45:
                mc = 4;
                nc = 5;
                break;
            case 0x54:
                mc = 5;
                nc = 4;
                break;
            case 0x44:
                mc = 4;
                nc = 4;
                break;
            case 0x53:
                mc = 5;
                nc = 3;
                break;
            case 0x35:
                mc = 3;
                nc = 5;
                break;
            case 0x43:
                mc = 4;
                nc = 3;
                break;
            case 0x34:
                mc = 3;
                nc = 4;
                break;
            case 0x52:
                mc = 5;
                nc = 2;
                break;
            case 0x33:
                mc = 3;
                nc = 3;
                break;
            case 0x25:
                mc = 2;
                nc = 5;
                break;
            case 0x42:
                mc = 4;
                nc = 2;
                break;
            case 0x24:
                mc = 2;
                nc = 4;
                break;
            case 0x32:
                mc = 3;
                nc = 2;
                break;
            case 0x23:
                mc = 2;
                nc = 3;
                break;
            case 0x51:
                mc = 5;
                nc = 1;
                break;
            case 0x41:
                mc = 4;
                nc = 1;
                break;
            case 0x22:
                mc = 2;
                nc = 2;
                break;
            case 0x15:
                mc = 1;
                nc = 5;
                break;
            case 0x14:
                mc = 1;
                nc = 4;
                break;
            case 0x31:
                mc = 3;
                nc = 1;
                break;
            case 0x13:
                mc = 1;
                nc = 3;
                break;
            case 0x21:
                mc = 2;
                nc = 1;
                break;
            case 0x12:
                mc = 1;
                nc = 2;
                break;
            case 0x11:
                mc = 1;
                nc = 1;
                break;
            default:
                return;
    }

    // If AVX512 is not supported, we can't use > 4x4 blocks
    /*if (((params.flags & HAS_AVX2) == 0 || (params.flags & IS_M_SERIES_MAC) == 0) && mc >= 4 && nc >= 4) {
        mc = 4;
        nc = 4;
    }*/

    gemmPtr(m0, m, n0, n, mc, nc, params);

    mp = m0 + (m - m0) / mc * mc;
    np = n0 + (n - n0) / nc * nc;
    gemm(mp, m, n0, np, gemmPtr, params);
    gemm(m0, mp, np, n, gemmPtr, params);
    gemm(mp, m, np, n, gemmPtr, params);
}

#if defined(__ARM_NEON__)
void __attribute__((noinline)) gemm_q8_q4_128_arm(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    // Mask to keep the first 4 bits of each byte
    int8x16_t mask_first_4bits = vdupq_n_u8(0x0f);
    //Subtract 8 from each byte to get signed values
    int8x16_t eight = vdupq_n_s8(0x8);
    int numBlocks = params.k / Q4_BLOCK_SIZE;

    __attribute__((aligned(16))) float scalef[4];

    // This fits on the stack (max of 5x5)
    for (int job = 0; job < tiles; ++job) {

        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        float32x4_t sums[RM][RN];

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = vdupq_n_f32(0.0f);
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;

            for (int i = 0; i < numBlocks; i += 4) { //128bits == 4floats
                int remainingBlocks = MIN(4, numBlocks - i);
                int aoo = ao;
                int boo = bo;

                for (int mi = 0; mi < RM; ++mi) {
                    ao = aoo;
                    bo = boo;

                    // Load float32
                    for (int sf = 0; sf < remainingBlocks; sf++) {
                        scalef[sf] = params.af[params.ldaf * (ii + mi) + ((ao + sf * Q4_BLOCK_SIZE) / Q4_BLOCK_SIZE)]
                                * params.bf[params.ldbf * (jj + ni) + (((bo + sf * (Q4_BLOCK_SIZE / 2)) * 2) / Q4_BLOCK_SIZE)];
                    }

                    for(int j = 0; j < remainingBlocks; j++, ao += 32, bo += 16) {
                        // Load 4 bytes into a 128-bit integer register
                        int8x16_t int_va0 = vld1q_s8((const signed char *)(params.a + params.lda * (ii + mi) + ao));
                        int8x16_t int_va1 = vld1q_s8((const signed char *)(params.a + params.lda * (ii + mi) + ao + 16));

                        // Load 8 bytes into a 128-bit integer register
                        int8x16_t int_vb0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8((const unsigned char *)(params.b + params.ldb * (jj + ni) + bo)),
                                            mask_first_4bits)), eight);

                        int8x16_t int_vb1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8((const unsigned char *)(params.b + params.ldb * (jj + ni) + bo)), 4)), eight);

                        sums[mi][ni] = vmlaq_n_f32(sums[mi][ni],
                            vcvtq_f32_s32(
                                vdotq_s32(
                                    vdotq_s32(vdupq_n_s32(0), int_va0, int_vb0),
                                    int_va1, int_vb1)), scalef[j]);
                    }
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = vaddvq_f32(sums[mi][ni]);
            }
        }
    }
}
#else
void __attribute__((noinline)) gemm_q8_q4_256(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    // Mask to keep the first 4 bits of each byte
    __m256i mask_first_4bits = _mm256_set1_epi8(0xF);
    //Subtract 8 from each byte to get signed values
    __m256i eight = _mm256_set1_epi8(8);
    int numBlocks = params.k / Q4_BLOCK_SIZE;

    // This fits on the stack (max of 5x5)
    __attribute__((aligned(64))) float scalef[8];
    for (int job = 0; job < tiles; ++job) {

        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        __attribute__((aligned(64))) __m256 sums[RN][RM];

        //Reset the sums to zero for this tile
        for (int i = 0; i < RN; i++) {
            for (int j = 0; j < RM; j++) {
                sums[i][j] = _mm256_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;

            for (int i = 0; i < numBlocks; i += 8) { //256bits == 8floats
                int aoo = ao;
                int boo = bo;

                for (int mi = 0; mi < RM; ++mi) {
                    ao = aoo;
                    bo = boo;

                    // Load float32
                     __m256 ablock = _mm256_loadu_ps(params.af + (params.ldaf * (ii + mi) + (ao / Q4_BLOCK_SIZE)));
                     __m256 bblock = _mm256_loadu_ps(params.bf + (params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)));
                     __m256 scaled = _mm256_mul_ps(ablock, bblock);
                     _mm256_store_ps(scalef, scaled);

                    for(int j = 0; j < 8; j++, ao += 32, bo += 16) {
                        // Load 16 bytes into 2 128-bit integer registers
                        __m256i int_va1 = _mm256_loadu_si256((__m256i const*)(params.a + params.lda * (ii + mi) + ao));
                        __m256i int_va0 = _mm256_sign_epi8(int_va1, int_va1);

                        // Load 8 bytes into a 128-bit integer register
                        __m128i int_vb0 = _mm_loadu_si128((__m128i const*)(params.b + params.ldb * (jj + ni) + bo)); // Load 128 bits

                        __m256i vb0 = _mm256_and_si256(mask_first_4bits,
                                                       _mm256_insertf128_si256(_mm256_castsi128_si256(int_vb0),
                                                                               _mm_srli_epi16(int_vb0, 4), 1));

                        vb0 = _mm256_sign_epi8(_mm256_sub_epi8(vb0, eight), int_va1);

                        __m256i res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(int_va0, vb0));
                        __m256 resf = _mm256_cvtepi32_ps(res);

                        // broadcast the float32 version of 'factor' to all elements
                        __m256 scale_f32 = _mm256_set1_ps(scalef[j]);

                        sums[ni][mi] = _mm256_fmadd_ps(scale_f32, resf, sums[ni][mi]);
                    }
                }
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            for (int mi = 0; mi < RM; ++mi) {

                __attribute__((aligned(64))) float result[8];
                _mm256_store_ps(result, sums[ni][mi]);

                float dot = 0.0;
                for(int i = 0; i < 8; ++i) {
                    dot += result[i];
                }
                //int idx = (params.ldc * (ii + mi)) + (jj + ni);
                //if (idx > params.roffset)
                //    fprintf(stderr, "ii: %d, ni: %d, jj: %d, mi: %d, ldc: %d, idx: %d, lim: %d\n", ii, ni, jj, mi, params.ldc,  idx, params.roffset);
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
}

void __attribute__((noinline)) gemm_q8_q4_512(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
#if defined(__AVX512F__)
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    // Mask to keep the first 4 bits of each byte
    __m256i mask_first_4bits = _mm256_set1_epi8(0xF);
    //Subtract 8 from each byte to get signed values
    __m256i eight = _mm256_set1_epi8(8);
    int numBlocks = params.k / Q4_BLOCK_SIZE;

    // This fits on the stack (max of 5x5)
    __attribute__((aligned(16))) float scalef[8];
    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        __m256 sums[RM][RN];

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm256_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;

            for (int i = 0; i < numBlocks; i += 8) { //256bits == 8floats
                int aoo = ao;
                int boo = bo;

                for (int mi = 0; mi < RM; ++mi) {
                    ao = aoo;
                    bo = boo;

                    // Load float32
                     __m256 ablock = _mm256_loadu_ps(params.af + (params.ldaf * (ii + mi) + (ao / Q4_BLOCK_SIZE)));
                     __m256 bblock = _mm256_loadu_ps(params.bf + (params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)));
                     __m256 scaled = _mm256_mul_ps(ablock, bblock);
                     _mm256_store_ps(scalef, scaled);

                    for(int j = 0; j < 8; j++, ao += 32, bo += 16) {
                        // Load 16 bytes into 2 128-bit integer registers
                        __m256i int_va1 = _mm256_loadu_si256((__m256i const*)(params.a + params.lda * (ii + mi) + ao));
                        __m256i int_va0 = _mm256_sign_epi8(int_va1, int_va1);

                        // Load 8 bytes into a 128-bit integer register
                        __m128i int_vb0 = _mm_loadu_si128((__m128i const*)(params.b + params.ldb * (jj + ni) + bo)); // Load 128 bits

                        __m256i vb0 = _mm256_and_si256(mask_first_4bits,
                                                       _mm256_insertf128_si256(_mm256_castsi128_si256(int_vb0),
                                                                               _mm_srli_epi16(int_vb0, 4), 1));

                        vb0 = _mm256_sign_epi8(_mm256_sub_epi8(vb0, eight), int_va1);

                        __m256i res;
                        #if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
                                res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), int_va0, vb0);
                        #else
                                res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(int_va0, vb0));
                        #endif

                        __m256 resf = _mm256_cvtepi32_ps(res);

                        // broadcast the float32 version of 'factor' to all elements
                        __m256 scale_f32 = _mm256_set1_ps(scalef[j]);

                        sums[mi][ni] = _mm256_fmadd_ps(scale_f32, resf, sums[mi][ni]);
                    }
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                float dot = _mm512_reduce_add_ps(_mm512_castps256_ps512(sums[mi][ni]));
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
#else
    gemm_q8_q4_256(m0, m, n0, n, RM, RN, params);
#endif
}
#endif //!ARM_NEON


void gemm_q8_q4(int flags, const float * restrict af, const char * restrict a, int aoffset, const float * restrict bf, const char* restrict b, int boffset, float * restrict r, int roffset, int m, int n0, int n, int k, int lda, int ldaf, int ldb, int ldbf, int ldc) {

    struct gemm_params p = {
                        .flags = flags,
                        .af = af,
                        .a = a,
                        .aoffset = aoffset,
                        .bf = bf,
                        .b = b,
                        .boffset = boffset,
                        .r = r,
                        .roffset = roffset,
                        .m = m,
                        .n = n,
                        .k = k,
                        .ldaf = ldaf,
                        .ldbf = ldbf,
                        .lda = lda,
                        .ldb = ldb,
                        .ldc = ldc
    };

    //fprintf(stderr, "m: %d, n0: %d, n: %d, k: %d, lda: %d, ldaf: %d, ldb: %d, ldbf: %d, ldc: %d\n", m, n0, n, k, lda, ldaf, ldb, ldbf, ldc);

#if !defined(__ARM_NEON__)
    ((flags & HAS_AVX2) != 0)
           ? gemm(0, m, n0, n0 + n, gemm_q8_q4_512, p)
           : gemm(0, m, n0, n0 + n, gemm_q8_q4_256, p);
#else
    gemm(0, m, n0, n0 + n, gemm_q8_q4_128_arm, p);
#endif
}

void gemm_q8_q4_batch(int flags, int batch_num, const float *af, const char *a, int aoffset, const float **bf, const char **b, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldaf, int ldb, int ldbf, int ldc)
{
    for (int i = 0; i < batch_num; i++) {
        gemm_q8_q4(flags, af, a, aoffset, bf[i], b[i], boffset, r[i], roffset, m, n0, n, k, lda, ldaf, ldb, ldbf, ldc);
    }
}

#if defined(__ARM_NEON__)
void gemm_f32_128_arm(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    // This fits on the stack (max of 5x5)
    float32x4_t sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = vdupq_n_f32(0.0f);
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 4, ao += 4, bo += 4) { // 128bits == 4floats
                // Load float32
                float32x4_t vb = vld1q_f32(params.bf + params.ldb * (jj + ni) + bo);

                for (int mi = 0; mi < RM; ++mi) {
                    float32x4_t va = vld1q_f32(params.af + params.lda * (ii + mi) + ao);

                    // Multiply and accumulate
                    sums[mi][ni] = vmlaq_f32(sums[mi][ni], va, vb);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = vaddvq_f32(sums[mi][ni]);
            }
        }
    }
}

#else
void __attribute__((noinline)) gemm_f32_256(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    // This fits on the stack (max of 5x5)
    __m256 sums[RN][RM] __attribute__((aligned(64)));

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RN; i++) {
            for (int j = 0; j < RM; j++) {
                sums[i][j] = _mm256_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 8, ao += 8, bo += 8) { // 256bits == 8floats
                // Load float32
                __m256 vb = _mm256_loadu_ps(params.bf + params.ldb * (jj + ni) + bo);

                for (int mi = 0; mi < RM; ++mi) {
                    __m256 va = _mm256_loadu_ps(params.af + params.lda * (ii + mi) + ao);

                    // Multiply and accumulate
                    sums[ni][mi] = _mm256_fmadd_ps(va, vb, sums[ni][mi]);
                }
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            for (int mi = 0; mi < RM; ++mi) {
                // Horizontal sum of the vector to get dot product
                float result[8] __attribute__((aligned(64)));
                _mm256_store_ps(result, sums[ni][mi]);

                float dot = 0.0;
                for(int i = 0; i < 8; ++i) {
                    dot += result[i];
                }
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
}

void gemm_f32_512(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
#if defined(__AVX512F__)
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    // This fits on the stack (max of 5x5)
    __m512 sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm512_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 16, ao += 16, bo += 16) { // 512bits == 16floats
                // Load float32
                __m512 vb = _mm512_loadu_ps(params.bf + params.ldb * (jj + ni) + bo);

                for (int mi = 0; mi < RM; ++mi) {
                    __m512 va = _mm512_loadu_ps(params.af + params.lda * (ii + mi) + ao);

                    // Multiply and accumulate
                    sums[mi][ni] = _mm512_fmadd_ps(va, vb, sums[mi][ni]);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                float r = _mm512_reduce_add_ps(sums[mi][ni]);
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = r;
            }
        }
    }
#else
    gemm_f32_256(m0, m, n0, n, RM, RN, params);
#endif
}
#endif //!ARM_NEON

void gemm_f32(int flags, const float *a, int aoffset, const float *b, int boffset, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
{
    struct gemm_params p = {
                        .flags = flags,
                        .af = a,
                        .a = NULL,
                        .aoffset = aoffset,
                        .bf = b,
                        .b = NULL,
                        .boffset = boffset,
                        .r = r,
                        .roffset = roffset,
                        .m = m,
                        .n = n,
                        .k = k,
                        .ldaf = 0,
                        .ldbf = 0,
                        .lda = lda,
                        .ldb = ldb,
                        .ldc = ldc
    };

#if !defined(__ARM_NEON__)
    ((flags & HAS_AVX2) != 0)
           ? gemm(0, m, n0, n0 + n, gemm_f32_512, p)
           : gemm(0, m, n0, n0 + n, gemm_f32_256, p);
#else
    gemm(0, m, n0, n0 + n, gemm_f32_128_arm, p);
#endif
}

void gemm_f32_batch(int flags, int batch_num, const float *a, int aoffset, const float **b, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
{
    for (int i = 0; i < batch_num; i++) {
        gemm_f32(flags, a, aoffset, b[i], boffset, r[i], roffset, m, n0, n, k, lda, ldb, ldc);
    }
}


#if defined(__ARM_NEON__)
void __attribute__((noinline)) gemm_f32_q4_128_arm(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    // Mask to keep the first 4 bits of each byte
    int8x16_t mask_first_4bits = vdupq_n_u8(0x0f);
    //Subtract 8 from each byte to get signed values
    int8x16_t eight = vdupq_n_s8(0x8);
    int numBlocks = params.k / Q4_BLOCK_SIZE;

    __attribute__((aligned(16))) float scalef[4];

    // This fits on the stack (max of 5x5)
    for (int job = 0; job < tiles; ++job) {

        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        float32x4_t sums[RM][RN];

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = vdupq_n_f32(0.0f);
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;

            for (int i = 0; i < numBlocks; i += 4) { //128bits == 4floats
                int aoo = ao;
                int boo = bo;

                for (int mi = 0; mi < RM; ++mi) {
                    ao = aoo;
                    bo = boo;

                    // Load float32
                    float32x4_t bblock = vld1q_f32(params.bf + (params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)));
                    vst1q_f32(scalef, bblock);

                    for(int j = 0; j < 4; j++, ao += 32, bo += 16) {
                        float32x4_t vb_f32 = vdupq_n_f32(scalef[j]);

                        // Load 4 bytes into a 128-bit integer register
                        float32x4_t f_va0 = vld1q_f32(params.af + params.lda * (ii + mi) + ao);
                        float32x4_t f_va1 = vld1q_f32(params.af + params.lda * (ii + mi) + ao + 4);
                        float32x4_t f_va2 = vld1q_f32(params.af + params.lda * (ii + mi) + ao + 8);
                        float32x4_t f_va3 = vld1q_f32(params.af + params.lda * (ii + mi) + ao + 12);

                        float32x4_t f_va4 = vld1q_f32(params.af + params.lda * (ii + mi) + ao + 16);
                        float32x4_t f_va5 = vld1q_f32(params.af + params.lda * (ii + mi) + ao + 20);
                        float32x4_t f_va6 = vld1q_f32(params.af + params.lda * (ii + mi) + ao + 24);
                        float32x4_t f_va7 = vld1q_f32(params.af + params.lda * (ii + mi) + ao + 28);

                        // Load 8 bytes into a 128-bit integer register
                        int8x16_t int_vb0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8((const unsigned char *)(params.b + params.ldb * (jj + ni) + bo)),
                                            mask_first_4bits)), eight);

                        // Convert int_vb0 into two float32x4_t registers
                        int16x8_t int_vb0_low = vmovl_s8(vget_low_s8(int_vb0));
                        int16x8_t int_vb0_high = vmovl_s8(vget_high_s8(int_vb0));
                        float32x4_t f_vb0_0 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_low_s16(int_vb0_low))));
                        float32x4_t f_vb0_1 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_high_s16(int_vb0_low))));
                        float32x4_t f_vb0_2 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_low_s16(int_vb0_high))));
                        float32x4_t f_vb0_3 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_high_s16(int_vb0_high))));

                        int8x16_t int_vb1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8((const unsigned char *)(params.b + params.ldb * (jj + ni) + bo)), 4)), eight);

                        // Convert int_vb0 into two float32x4_t registers
                        int16x8_t int_vb1_low = vmovl_s8(vget_low_s8(int_vb1));
                        int16x8_t int_vb1_high = vmovl_s8(vget_high_s8(int_vb1));
                        float32x4_t f_vb1_0 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_low_s16(int_vb1_low))));
                        float32x4_t f_vb1_1 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_high_s16(int_vb1_low))));
                        float32x4_t f_vb1_2 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_low_s16(int_vb1_high))));
                        float32x4_t f_vb1_3 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_high_s16(int_vb1_high))));

                        // FMA operations for sums[mi][ni] with each of the 8 pairs of va and vb
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va0, f_vb0_0);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va1, f_vb0_1);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va2, f_vb0_2);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va3, f_vb0_3);

                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va4, f_vb1_0);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va5, f_vb1_1);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va6, f_vb1_2);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va7, f_vb1_3);
                    }
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = vaddvq_f32(sums[mi][ni]);
            }
        }
    }
}
#else
void gemm_f32_q4_256(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    // Mask to keep the first 4 bits of each byte
    __m128i mask_first_4bits = _mm_set1_epi8(0xF);
    //Subtract 8 from each byte to get signed values
    __m128i eight = _mm_set1_epi8(8);

    // This fits on the stack (max of 5x5)
    __m256 sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm256_setzero_ps();
            }
        }

        for(int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;

            for(int j = 0; j < params.k; j += 32, ao += 32, bo += 16) {
                for (int mi = 0; mi < RM; ++mi) {
                        // Load float32
                        __m256 va0 = _mm256_loadu_ps(params.af + params.lda * (ii + mi) + ao);
                        __m256 va1 = _mm256_loadu_ps(params.af + params.lda * (ii + mi) + ao + 8);
                        __m256 va2 = _mm256_loadu_ps(params.af + params.lda * (ii + mi) + ao + 8 + 8);
                        __m256 va3 = _mm256_loadu_ps(params.af + params.lda * (ii + mi) + ao + 8 + 8 + 8);

                        // Load float32
                        float bfactor = params.bf[params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)];

                        // broadcast the float32 version of 'factor' to all elements
                        __m256 vb_f32 = _mm256_set1_ps(bfactor);

                        // Load 8 bytes into a 128-bit integer register
                        __m128i int_vb0 = _mm_loadl_epi64((__m128i const*)(params.b + params.ldb * (jj + ni) + bo)); // Load lower 64 bits
                        __m128i int_vb1 = _mm_loadl_epi64((__m128i const*)(params.b + params.ldb * (jj + ni) + bo + 8)); // Load lower 64 bits

                        // Masked values
                        __m128i first_4bits0 = _mm_and_si128(int_vb0, mask_first_4bits);
                        __m128i first_4bits1 = _mm_and_si128(int_vb1, mask_first_4bits);

                        // Shift first 4 bits to rightmost positions
                        __m128i last_4bits0 = _mm_srli_epi16(int_vb0, 4);
                        __m128i last_4bits1 = _mm_srli_epi16(int_vb1, 4);

                        last_4bits0 = _mm_and_si128(last_4bits0, mask_first_4bits);
                        last_4bits1 = _mm_and_si128(last_4bits1, mask_first_4bits);

                        //Subtract 8 from each int
                        first_4bits0 = _mm_sub_epi8(first_4bits0, eight);
                        first_4bits1 = _mm_sub_epi8(first_4bits1, eight);

                        last_4bits0 = _mm_sub_epi8(last_4bits0, eight);
                        last_4bits1 = _mm_sub_epi8(last_4bits1, eight);

                        // Extend these bytes to 32-bit integers (low and high)
                        __m256i int_vb_ext_lo0 = _mm256_cvtepi8_epi32(first_4bits0);
                        __m256i int_vb_ext_lo1 = _mm256_cvtepi8_epi32(first_4bits1);

                        __m256i int_vb_ext_hi0 = _mm256_cvtepi8_epi32(last_4bits0);
                        __m256i int_vb_ext_hi1 = _mm256_cvtepi8_epi32(last_4bits1);

                        // Convert these 32-bit integers to floats
                        __m256 float_vb_lo0 = _mm256_cvtepi32_ps(int_vb_ext_lo0);
                        __m256 float_vb_lo1 = _mm256_cvtepi32_ps(int_vb_ext_lo1);

                        __m256 float_vb_hi0 = _mm256_cvtepi32_ps(int_vb_ext_hi0);
                        __m256 float_vb_hi1 = _mm256_cvtepi32_ps(int_vb_ext_hi1);

                        // Perform the scaling
                        __m256 vb_scaled_lo0 = _mm256_mul_ps(vb_f32, float_vb_lo0);
                        __m256 vb_scaled_lo1 = _mm256_mul_ps(vb_f32, float_vb_lo1);
                        __m256 vb_scaled_hi0 = _mm256_mul_ps(vb_f32, float_vb_hi0);
                        __m256 vb_scaled_hi1 = _mm256_mul_ps(vb_f32, float_vb_hi1);

                        // Multiply and accumulate
                        sums[mi][ni] = _mm256_fmadd_ps(va0, vb_scaled_lo0, sums[mi][ni]);
                        sums[mi][ni] = _mm256_fmadd_ps(va1, vb_scaled_lo1, sums[mi][ni]);
                        sums[mi][ni] = _mm256_fmadd_ps(va2, vb_scaled_hi0, sums[mi][ni]);
                        sums[mi][ni] = _mm256_fmadd_ps(va3, vb_scaled_hi1, sums[mi][ni]);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                __attribute__((aligned(16))) float result[8];
                _mm256_store_ps(result, sums[mi][ni]);

                float dot = 0.0;
                for(int i = 0; i < 8; ++i) {
                    dot += result[i];
                }
                //if (params.roffset > 0)
                //    fprintf(stderr, "ii: %d, ni: %d, jj: %d, mi: %d, ldc: %d, roffset: %d\n", ii, ni, jj, mi, params.ldc, params.roffset);
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
}

void gemm_f32_q4_512(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
#if defined(__AVX512F__)
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    // Mask to keep the first 4 bits of each byte
    __m128i mask_first_4bits = _mm_set1_epi8(0xF);
    //Subtract 8 from each byte to get signed values
    __m128i eight = _mm_set1_epi8(8);
    //int numBlocks = params.k / Q4_BLOCK_SIZE;

    // This fits on the stack (max of 5x5)
    __m512 sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm512_setzero_ps();
            }
        }

        for(int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;

            for(int j = 0; j < params.k; j += 32, ao += 32, bo += 16) {
                for (int mi = 0; mi < RM; ++mi) {
                        // Load float32
                        __m512 va0 = _mm512_loadu_ps(params.af + params.lda * (ii + mi) + ao);
                        __m512 va1 = _mm512_loadu_ps(params.af + params.lda * (ii + mi) + ao + 16);

                        // Load float32
                        float bfactor = params.bf[params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)];

                        // broadcast the float32 version of 'factor' to all elements
                        __m512 vb_f32 = _mm512_set1_ps(bfactor);

                        // Load 8 bytes into a 128-bit integer register
                        __m128i int_vb0 = _mm_loadu_si128((__m128i const*)(params.b + params.ldb * (jj + ni) + bo)); // Load 128 bits

                        // Masked values
                        __m128i first_4bits0 = _mm_and_si128(int_vb0, mask_first_4bits);

                        // Shift first 4 bits to rightmost positions
                        __m128i last_4bits0 = _mm_srli_epi16(int_vb0, 4);
                        last_4bits0 = _mm_and_si128(last_4bits0, mask_first_4bits);

                        //Subtract 8 from each int
                        first_4bits0 = _mm_sub_epi8(first_4bits0, eight);
                        last_4bits0 = _mm_sub_epi8(last_4bits0, eight);

                        // Extend these bytes to 32-bit integers (low and high)
                        __m512i int_vb_ext_lo0 = _mm512_cvtepi8_epi32(first_4bits0);
                        __m512i int_vb_ext_hi0 = _mm512_cvtepi8_epi32(last_4bits0);

                        __m512 float_vb_lo0 = _mm512_cvtepi32_ps(int_vb_ext_lo0);
                        __m512 float_vb_hi0 = _mm512_cvtepi32_ps(int_vb_ext_hi0);

                        // Perform the scaling
                        __m512 vb_scaled_lo0 = _mm512_mul_ps(vb_f32, float_vb_lo0);
                        __m512 vb_scaled_hi0 = _mm512_mul_ps(vb_f32, float_vb_hi0);

                        // Multiply and accumulate
                        sums[mi][ni] = _mm512_fmadd_ps(va0, vb_scaled_lo0, sums[mi][ni]);
                        sums[mi][ni] = _mm512_fmadd_ps(va1, vb_scaled_hi0, sums[mi][ni]);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                float r = _mm512_reduce_add_ps(sums[mi][ni]);
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = r;
            }
        }
   }

#else
    gemm_f32_q4_256(m0, m, n0, n, RM, RN, params);
#endif
}
#endif //!ARM_NEON

static void gemm_f32_q4_scalar(const float *a, int aoffset, const float *bf, const char* b, int boffset, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc) {
    for (int row = 0; row < m; row++) {
        for (int out_col = 0; out_col < n; out_col++) {
            int weight_row = n0 + out_col;
            float sum = 0.0f;
            for (int col = 0; col < k; col++) {
                int logical_col = (boffset * 2) + col;
                int within_block = logical_col % Q4_BLOCK_SIZE;
                size_t byte_index = ((size_t) weight_row * (size_t) ldb)
                        + ((size_t) logical_col / Q4_BLOCK_SIZE) * (Q4_BLOCK_SIZE / 2)
                        + (within_block % (Q4_BLOCK_SIZE / 2));
                unsigned char packed = (unsigned char) b[byte_index];
                int nibble = within_block < (Q4_BLOCK_SIZE / 2) ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                int q = nibble - 8;
                float scale = bf[(size_t) weight_row * (size_t) ldbf + ((size_t) logical_col / Q4_BLOCK_SIZE)];
                sum += a[row * lda + aoffset + col] * q * scale;
            }
            ptrdiff_t r_index = (ptrdiff_t) row * (ptrdiff_t) ldc + (ptrdiff_t) weight_row - (ptrdiff_t) roffset;
            r[r_index] = sum;
        }
    }
}

void gemm_f32_q4(int flags, const float *a, int aoffset, const float *bf, const char* b, int boffset, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc)
{
    gemm_f32_q4_scalar(a, aoffset, bf, b, boffset, r, roffset, m, n0, n, k, lda, ldb, ldbf, ldc);
    return;

    struct gemm_params p = {
                        .flags = flags,
                        .af = a,
                        .a = NULL,
                        .aoffset = aoffset,
                        .bf = bf,
                        .b = b,
                        .boffset = boffset,
                        .r = r,
                        .roffset = roffset,
                        .m = m,
                        .n = n,
                        .k = k,
                        .ldaf = 0,
                        .ldbf = ldbf,
                        .lda = lda,
                        .ldb = ldb,
                        .ldc = ldc
    };

#if !defined(__ARM_NEON__)
    ((flags & HAS_AVX2) != 0)
           ? gemm(0, m, n0, n0 + n, gemm_f32_q4_512, p)
           : gemm(0, m, n0, n0 + n, gemm_f32_q4_256, p);
#else
    gemm(0, m, n0, n0 + n, gemm_f32_q4_128_arm, p);
#endif
}

void gemm_f32_q4_batch(int flags, int batch_num, const float *a, int aoffset, const float **bf, const char **b, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc)
{
    for (int i = 0; i < batch_num; i++) {
        gemm_f32_q4(flags, a, aoffset, bf[i], b[i], boffset, r[i], roffset, m, n0, n, k, lda, ldb, ldbf, ldc);
    }
}

static void gemm_bf16_q4_scalar(const short *a, int aoffset, const float *bf, const char* b, int boffset, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc) {
    for (int row = 0; row < m; row++) {
        for (int out_col = 0; out_col < n; out_col++) {
            int weight_row = n0 + out_col;
            float sum = 0.0f;
            for (int col = 0; col < k; col++) {
                int logical_col = (boffset * 2) + col;
                int within_block = logical_col % Q4_BLOCK_SIZE;
                size_t byte_index = ((size_t) weight_row * (size_t) ldb)
                        + ((size_t) logical_col / Q4_BLOCK_SIZE) * (Q4_BLOCK_SIZE / 2)
                        + (within_block % (Q4_BLOCK_SIZE / 2));
                unsigned char packed = (unsigned char) b[byte_index];
                int nibble = within_block < (Q4_BLOCK_SIZE / 2) ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
                int q = nibble - 8;
                float scale = bf[(size_t) weight_row * (size_t) ldbf + ((size_t) logical_col / Q4_BLOCK_SIZE)];
                sum += bf16_to_fp32(a[row * lda + aoffset + col]) * q * scale;
            }
            ptrdiff_t r_index = (ptrdiff_t) row * (ptrdiff_t) ldc + (ptrdiff_t) weight_row - (ptrdiff_t) roffset;
            r[r_index] = sum;
        }
    }
}

#if defined(__ARM_NEON__)
void __attribute__((noinline)) gemm_bf16_q4_128_arm(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    int8x16_t mask_first_4bits = vdupq_n_u8(0x0f);
    int8x16_t eight = vdupq_n_s8(0x8);
    __attribute__((aligned(16))) float scalef[4];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;
        float32x4_t sums[RM][RN];
        for (int i = 0; i < RM; i++) for (int j = 0; j < RN; j++) sums[i][j] = vdupq_n_f32(0.0f);

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            int numBlocks = params.k / Q4_BLOCK_SIZE;
            for (int i = 0; i < numBlocks; i += 4) {
                int remainingBlocks = MIN(4, numBlocks - i);
                int aoo = ao;
                int boo = bo;
                for (int mi = 0; mi < RM; ++mi) {
                    ao = aoo;
                    bo = boo;
                    for (int sf = 0; sf < remainingBlocks; sf++) {
                        scalef[sf] = params.bf[params.ldbf * (jj + ni) + (((bo + sf * (Q4_BLOCK_SIZE / 2)) * 2) / Q4_BLOCK_SIZE)];
                    }
                    for(int j = 0; j < remainingBlocks; j++, ao += 32, bo += 16) {
                        float32x4_t vb_f32 = vdupq_n_f32(scalef[j]);
                        float32x4_t f_va0 = load_bf16x4_as_f32(params.as + params.lda * (ii + mi) + ao);
                        float32x4_t f_va1 = load_bf16x4_as_f32(params.as + params.lda * (ii + mi) + ao + 4);
                        float32x4_t f_va2 = load_bf16x4_as_f32(params.as + params.lda * (ii + mi) + ao + 8);
                        float32x4_t f_va3 = load_bf16x4_as_f32(params.as + params.lda * (ii + mi) + ao + 12);
                        float32x4_t f_va4 = load_bf16x4_as_f32(params.as + params.lda * (ii + mi) + ao + 16);
                        float32x4_t f_va5 = load_bf16x4_as_f32(params.as + params.lda * (ii + mi) + ao + 20);
                        float32x4_t f_va6 = load_bf16x4_as_f32(params.as + params.lda * (ii + mi) + ao + 24);
                        float32x4_t f_va7 = load_bf16x4_as_f32(params.as + params.lda * (ii + mi) + ao + 28);

                        int8x16_t int_vb0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8((const unsigned char *)(params.b + params.ldb * (jj + ni) + bo)), mask_first_4bits)), eight);
                        int8x16_t int_vb1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8((const unsigned char *)(params.b + params.ldb * (jj + ni) + bo)), 4)), eight);
                        int16x8_t int_vb0_low = vmovl_s8(vget_low_s8(int_vb0));
                        int16x8_t int_vb0_high = vmovl_s8(vget_high_s8(int_vb0));
                        int16x8_t int_vb1_low = vmovl_s8(vget_low_s8(int_vb1));
                        int16x8_t int_vb1_high = vmovl_s8(vget_high_s8(int_vb1));
                        float32x4_t f_vb0_0 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_low_s16(int_vb0_low))));
                        float32x4_t f_vb0_1 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_high_s16(int_vb0_low))));
                        float32x4_t f_vb0_2 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_low_s16(int_vb0_high))));
                        float32x4_t f_vb0_3 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_high_s16(int_vb0_high))));
                        float32x4_t f_vb1_0 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_low_s16(int_vb1_low))));
                        float32x4_t f_vb1_1 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_high_s16(int_vb1_low))));
                        float32x4_t f_vb1_2 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_low_s16(int_vb1_high))));
                        float32x4_t f_vb1_3 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_high_s16(int_vb1_high))));
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va0, f_vb0_0);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va1, f_vb0_1);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va2, f_vb0_2);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va3, f_vb0_3);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va4, f_vb1_0);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va5, f_vb1_1);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va6, f_vb1_2);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va7, f_vb1_3);
                    }
                }
            }
        }
        for (int mi = 0; mi < RM; ++mi) for (int ni = 0; ni < RN; ++ni) params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = vaddvq_f32(sums[mi][ni]);
    }
}
#else
void gemm_bf16_q4_256(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    __m128i mask_first_4bits = _mm_set1_epi8(0xF);
    __m128i eight = _mm_set1_epi8(8);
    __m256 sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;
        for (int i = 0; i < RM; i++) for (int j = 0; j < RN; j++) sums[i][j] = _mm256_setzero_ps();
        for(int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 32, ao += 32, bo += 16) {
                for (int mi = 0; mi < RM; ++mi) {
                    __m256 va0 = load_bf16x8_as_f32(params.as + params.lda * (ii + mi) + ao);
                    __m256 va1 = load_bf16x8_as_f32(params.as + params.lda * (ii + mi) + ao + 8);
                    __m256 va2 = load_bf16x8_as_f32(params.as + params.lda * (ii + mi) + ao + 16);
                    __m256 va3 = load_bf16x8_as_f32(params.as + params.lda * (ii + mi) + ao + 24);
                    float bfactor = params.bf[params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)];
                    __m256 vb_f32 = _mm256_set1_ps(bfactor);
                    __m128i int_vb0 = _mm_loadl_epi64((__m128i const*)(params.b + params.ldb * (jj + ni) + bo));
                    __m128i int_vb1 = _mm_loadl_epi64((__m128i const*)(params.b + params.ldb * (jj + ni) + bo + 8));
                    __m128i first_4bits0 = _mm_sub_epi8(_mm_and_si128(int_vb0, mask_first_4bits), eight);
                    __m128i first_4bits1 = _mm_sub_epi8(_mm_and_si128(int_vb1, mask_first_4bits), eight);
                    __m128i last_4bits0 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(int_vb0, 4), mask_first_4bits), eight);
                    __m128i last_4bits1 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(int_vb1, 4), mask_first_4bits), eight);
                    __m256 float_vb_lo0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(first_4bits0));
                    __m256 float_vb_lo1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(first_4bits1));
                    __m256 float_vb_hi0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(last_4bits0));
                    __m256 float_vb_hi1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(last_4bits1));
                    sums[mi][ni] = _mm256_fmadd_ps(va0, _mm256_mul_ps(vb_f32, float_vb_lo0), sums[mi][ni]);
                    sums[mi][ni] = _mm256_fmadd_ps(va1, _mm256_mul_ps(vb_f32, float_vb_lo1), sums[mi][ni]);
                    sums[mi][ni] = _mm256_fmadd_ps(va2, _mm256_mul_ps(vb_f32, float_vb_hi0), sums[mi][ni]);
                    sums[mi][ni] = _mm256_fmadd_ps(va3, _mm256_mul_ps(vb_f32, float_vb_hi1), sums[mi][ni]);
                }
            }
        }
        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                __attribute__((aligned(16))) float result[8];
                _mm256_store_ps(result, sums[mi][ni]);
                float dot = 0.0;
                for(int i = 0; i < 8; ++i) dot += result[i];
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
}

void gemm_bf16_q4_512(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
#if defined(__AVX512F__)
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    __m128i mask_first_4bits = _mm_set1_epi8(0xF);
    __m128i eight = _mm_set1_epi8(8);
    __m512 sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;
        for (int i = 0; i < RM; i++) for (int j = 0; j < RN; j++) sums[i][j] = _mm512_setzero_ps();
        for(int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 32, ao += 32, bo += 16) {
                for (int mi = 0; mi < RM; ++mi) {
                    __m512 va0 = load_bf16x16_as_f32(params.as + params.lda * (ii + mi) + ao);
                    __m512 va1 = load_bf16x16_as_f32(params.as + params.lda * (ii + mi) + ao + 16);
                    float bfactor = params.bf[params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)];
                    __m512 vb_f32 = _mm512_set1_ps(bfactor);
                    __m128i int_vb0 = _mm_loadu_si128((__m128i const*)(params.b + params.ldb * (jj + ni) + bo));
                    __m128i first_4bits0 = _mm_sub_epi8(_mm_and_si128(int_vb0, mask_first_4bits), eight);
                    __m128i last_4bits0 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(int_vb0, 4), mask_first_4bits), eight);
                    __m512 float_vb_lo0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(first_4bits0));
                    __m512 float_vb_hi0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(last_4bits0));
                    sums[mi][ni] = _mm512_fmadd_ps(va0, _mm512_mul_ps(vb_f32, float_vb_lo0), sums[mi][ni]);
                    sums[mi][ni] = _mm512_fmadd_ps(va1, _mm512_mul_ps(vb_f32, float_vb_hi0), sums[mi][ni]);
                }
            }
        }
        for (int mi = 0; mi < RM; ++mi) for (int ni = 0; ni < RN; ++ni) params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = _mm512_reduce_add_ps(sums[mi][ni]);
   }
#else
    gemm_bf16_q4_256(m0, m, n0, n, RM, RN, params);
#endif
}
#endif

void gemm_bf16_q4(int flags, const short *a, int aoffset, const float *bf, const char* b, int boffset, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc)
{
    if (m < 2 || n < 2 || ((aoffset | (boffset * 2) | k) & (Q4_BLOCK_SIZE - 1)) != 0) {
        gemm_bf16_q4_scalar(a, aoffset, bf, b, boffset, r, roffset, m, n0, n, k, lda, ldb, ldbf, ldc);
        return;
    }

    struct gemm_params p = {
                        .flags = flags,
                        .as = a,
                        .aoffset = aoffset,
                        .bf = bf,
                        .b = b,
                        .boffset = boffset,
                        .r = r,
                        .roffset = roffset,
                        .m = m,
                        .n = n,
                        .k = k,
                        .ldaf = 0,
                        .ldbf = ldbf,
                        .lda = lda,
                        .ldb = ldb,
                        .ldc = ldc
    };

#if !defined(__ARM_NEON__)
    ((flags & HAS_AVX2) != 0)
           ? gemm(0, m, n0, n0 + n, gemm_bf16_q4_512, p)
           : gemm(0, m, n0, n0 + n, gemm_bf16_q4_256, p);
#else
    gemm(0, m, n0, n0 + n, gemm_bf16_q4_128_arm, p);
#endif
}

void gemm_bf16_q4_batch(int flags, int batch_num, const short *a, int aoffset, const float **bf, const char **b, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc)
{
    for (int i = 0; i < batch_num; i++) {
        gemm_bf16_q4(flags, a, aoffset, bf[i], b[i], boffset, r[i], roffset, m, n0, n, k, lda, ldb, ldbf, ldc);
    }
}


///// GEMM BF16
#if defined(__ARM_NEON__)
void gemm_bf16_128_arm(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    // This fits on the stack (max of 5x5)
    float32x4_t sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = vdupq_n_f32(0.0f);
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 8, ao += 8, bo += 8) { // 128bits == 8bfloats
                // Load shorts
                uint16x8_t vb = vld1q_u16((const uint16_t*)(params.bs + params.ldb * (jj + ni) + bo));

                // Extract and convert to float
                uint32x4_t vb0i = vmovl_u16(vget_low_u16(vb));
                uint32x4_t vb1i = vmovl_u16(vget_high_u16(vb));
                float32x4_t vb0 = vreinterpretq_f32_u32(vshlq_n_u32(vb0i, 16));
                float32x4_t vb1 = vreinterpretq_f32_u32(vshlq_n_u32(vb1i, 16));

                for (int mi = 0; mi < RM; ++mi) {
                    uint16x8_t va = vld1q_u16((const uint16_t*)(params.as + params.lda * (ii + mi) + ao));

                    // Extract and convert to float
                    uint32x4_t va0i = vmovl_u16(vget_low_u16(va));
                    uint32x4_t va1i = vmovl_u16(vget_high_u16(va));
                    float32x4_t va0 = vreinterpretq_f32_u32(vshlq_n_u32(va0i, 16));
                    float32x4_t va1 = vreinterpretq_f32_u32(vshlq_n_u32(va1i, 16));

                    // Multiply and accumulate
                    sums[mi][ni] = vmlaq_f32(sums[mi][ni], va0, vb0);
                    sums[mi][ni] = vmlaq_f32(sums[mi][ni], va1, vb1);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                float dot = vaddvq_f32(sums[mi][ni]);

                if (params.rs != NULL)
                    params.rs[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = fp32_to_bf16(dot);
                else
                    params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
}

#else

void __attribute__((noinline)) gemm_bf16_256(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    float result[8] __attribute__((aligned(32)));

    // This fits on the stack (max of 5x5)
    __m256 sums[RN][RM];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RN; i++) {
            for (int j = 0; j < RM; j++) {
                sums[i][j] = _mm256_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 16, ao += 16, bo +=16) { // 256bits == 16bfloats
                // Load shorts
                __m256i vb = _mm256_loadu_si256((__m256i*)(params.bs + params.ldb * (jj + ni) + bo));

                // Extract lower 8 shorts and convert to int (lower 128 bits)
                __m256i vb0i = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(vb, 0));
                // Shift left 16 bits and convert to float
                __m256 vb0 = _mm256_castsi256_ps(_mm256_slli_epi32(vb0i, 16));

                // Extract lower 8 shorts and convert to int (upper 128 bits)
                __m256i vb1i = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(vb, 1));
                // Shift left 16 bits and convert to float
                __m256 vb1 = _mm256_castsi256_ps(_mm256_slli_epi32(vb1i, 16));

                for (int mi = 0; mi < RM; ++mi) {
                    // Load shorts
                    __m256i va = _mm256_loadu_si256((__m256i*)(params.as + params.lda * (ii + mi) + ao));

                    // Extract lower 8 shorts and convert to int (lower 128 bits)
                    __m256i va0i = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(va, 0));
                    // Shift left 16 bits and convert to float
                    __m256 va0 = _mm256_castsi256_ps(_mm256_slli_epi32(va0i, 16));

                    // Extract lower 8 shorts and convert to int (upper 128 bits)
                    __m256i va1i = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(va, 1));
                    // Shift left 16 bits and convert to float
                    __m256 va1 = _mm256_castsi256_ps(_mm256_slli_epi32(va1i, 16));

                    // Multiply and accumulate
                    sums[ni][mi] = _mm256_fmadd_ps(va0, vb0, sums[ni][mi]);
                    sums[ni][mi] = _mm256_fmadd_ps(va1, vb1, sums[ni][mi]);
                }
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            for (int mi = 0; mi < RM; ++mi) {
                // Horizontal sum of the vector to get dot product
                _mm256_store_ps(result, sums[ni][mi]);

                float dot = 0.0;
                for(int i = 0; i < 8; ++i) {
                    dot += result[i];
                }
                if (params.rs != NULL)
                    params.rs[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = fp32_to_bf16(dot);
                else
                    params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
}

void gemm_bf16_512(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
#if defined(__AVX512F__)
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    // This fits on the stack (max of 5x5)
    __m512 sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm512_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 32, ao += 32, bo += 32) { // 512bits == 32bfloats
                // Load shorts
                __m512i vb = _mm512_loadu_si512((__m512i*)(params.bs + params.ldb * (jj + ni) + bo));

                // Extract lower 8 shorts and convert to int (lower 128 bits)
                __m512i vb0i = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(vb, 0));
                // Shift left 16 bits and convert to float
                __m512 vb0 = _mm512_castsi512_ps(_mm512_slli_epi32(vb0i, 16));

                // Extract lower 8 shorts and convert to int (upper 128 bits)
                __m512i vb1i = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(vb, 1));
                // Shift left 16 bits and convert to float
                __m512 vb1 = _mm512_castsi512_ps(_mm512_slli_epi32(vb1i, 16));

                for (int mi = 0; mi < RM; ++mi) {
                    // Load shorts
                    __m512i va = _mm512_loadu_si512((__m512i*)(params.as + params.lda * (ii + mi) + ao));

                    // Extract lower 8 shorts and convert to int (lower 128 bits)
                    __m512i va0i = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(va, 0));
                    // Shift left 16 bits and convert to float
                    __m512 va0 = _mm512_castsi512_ps(_mm512_slli_epi32(va0i, 16));

                    // Extract lower 8 shorts and convert to int (upper 128 bits)
                    __m512i va1i = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(va, 1));
                    // Shift left 16 bits and convert to float
                    __m512 va1 = _mm512_castsi512_ps(_mm512_slli_epi32(va1i, 16));


                    // Multiply and accumulate
                    sums[mi][ni] = _mm512_fmadd_ps(va0, vb0, sums[mi][ni]);
                    sums[mi][ni] = _mm512_fmadd_ps(va1, vb1, sums[mi][ni]);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                float r = _mm512_reduce_add_ps(sums[mi][ni]);
                if (params.rs != NULL)
                    params.rs[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = fp32_to_bf16(r);
                else
                    params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = r;
            }
        }
    }
#else
    gemm_bf16_256(m0, m, n0, n, RM, RN, params);
#endif
}
#endif //!ARM_NEON

void gemm_bf16(int flags, const short *a, int aoffset, const short *b, int boffset, short *rs, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
{
    struct gemm_params p = {
                        .flags = flags,
                        .as = a,
                        .aoffset = aoffset,
                        .bs = b,
                        .boffset = boffset,
                        .rs = rs,
                        .r = r,
                        .roffset = roffset,
                        .m = m,
                        .n = n,
                        .k = k,
                        .ldaf = 0,
                        .ldbf = 0,
                        .lda = lda,
                        .ldb = ldb,
                        .ldc = ldc
    };

#if !defined(__ARM_NEON__)
    ((flags & HAS_AVX2) != 0)
           ? gemm(0, m, n0, n0 + n, gemm_bf16_512, p)
           : gemm(0, m, n0, n0 + n, gemm_bf16_256, p);
#else
    gemm(0, m, n0, n0 + n, gemm_bf16_128_arm, p);
#endif
}

void gemm_bf16_batch(int flags, int batch_num, const short *a, int aoffset, const short **b, int boffset, short **rs, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
{
    for (int i = 0; i < batch_num; i++) {
        gemm_bf16(flags, a, aoffset, b[i], boffset, rs != NULL ? rs[i] : NULL, r != NULL ? r[i] : NULL, roffset, m, n0, n, k, lda, ldb, ldc);
    }
}


///// GEMM F32 BF16
#if defined(__ARM_NEON__)
void gemm_f32_bf16_128_arm(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    // This fits on the stack (max of 5x5)
    float32x4_t sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = vdupq_n_f32(0.0f);
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 8, ao += 8, bo += 8) { // 128bits == 8bfloats
                // Load shorts
                uint16x8_t vb = vld1q_u16((const uint16_t*)(params.bs + params.ldb * (jj + ni) + bo));

                // Extract and convert to float
                uint32x4_t vb0i = vmovl_u16(vget_low_u16(vb));
                uint32x4_t vb1i = vmovl_u16(vget_high_u16(vb));
                float32x4_t vb0 = vreinterpretq_f32_u32(vshlq_n_u32(vb0i, 16));
                float32x4_t vb1 = vreinterpretq_f32_u32(vshlq_n_u32(vb1i, 16));

                for (int mi = 0; mi < RM; ++mi) {
                    float32x4_t va0 = vld1q_f32(params.af + params.lda * (ii + mi) + ao);
                    float32x4_t va1 = vld1q_f32(params.af + params.lda * (ii + mi) + ao + 4);

                    // Multiply and accumulate
                    sums[mi][ni] = vmlaq_f32(sums[mi][ni], va0, vb0);
                    sums[mi][ni] = vmlaq_f32(sums[mi][ni], va1, vb1);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product

                float dot = vaddvq_f32(sums[mi][ni]);
                if (params.rs != NULL)
                    params.rs[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = fp32_to_bf16(dot);
                else
                    params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
}

#else

void __attribute__((noinline)) gemm_f32_bf16_256(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    float result[8] __attribute__((aligned(32)));

    // This fits on the stack (max of 5x5)
    __m256 sums[RN][RM];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RN; i++) {
            for (int j = 0; j < RM; j++) {
                sums[i][j] = _mm256_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 16, ao += 16, bo +=16) { // 256bits == 16bfloats
                // Load shorts
                __m256i vb = _mm256_loadu_si256((__m256i*)(params.bs + params.ldb * (jj + ni) + bo));

                // Extract lower 8 shorts and convert to int (lower 128 bits)
                __m256i vb0i = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(vb, 0));
                // Shift left 16 bits and convert to float
                __m256 vb0 = _mm256_castsi256_ps(_mm256_slli_epi32(vb0i, 16));

                // Extract lower 8 shorts and convert to int (upper 128 bits)
                __m256i vb1i = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(vb, 1));
                // Shift left 16 bits and convert to float
                __m256 vb1 = _mm256_castsi256_ps(_mm256_slli_epi32(vb1i, 16));

                for (int mi = 0; mi < RM; ++mi) {
                    __m256 va0 = _mm256_loadu_ps(params.af + params.lda * (ii + mi) + ao);
                    __m256 va1 = _mm256_loadu_ps(params.af + params.lda * (ii + mi) + ao + 8);

                    // Multiply and accumulate
                    sums[ni][mi] = _mm256_fmadd_ps(va0, vb0, sums[ni][mi]);
                    sums[ni][mi] = _mm256_fmadd_ps(va1, vb1, sums[ni][mi]);
                }
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            for (int mi = 0; mi < RM; ++mi) {
                // Horizontal sum of the vector to get dot product
                _mm256_store_ps(result, sums[ni][mi]);

                float dot = 0.0;
                for(int i = 0; i < 8; ++i) {
                    dot += result[i];
                }
                if (params.rs != NULL)
                    params.rs[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = fp32_to_bf16(dot);
                else
                    params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
}

void gemm_f32_bf16_512(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
#if defined(__AVX512F__)
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    // This fits on the stack (max of 5x5)
    __m512 sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm512_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 32, ao += 32, bo += 32) { // 512bits == 32bfloats
                // Load shorts
                __m512i vb = _mm512_loadu_si512((__m512i*)(params.bs + params.ldb * (jj + ni) + bo));

                // Extract lower 8 shorts and convert to int (lower 128 bits)
                __m512i vb0i = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(vb, 0));
                // Shift left 16 bits and convert to float
                __m512 vb0 = _mm512_castsi512_ps(_mm512_slli_epi32(vb0i, 16));

                // Extract lower 8 shorts and convert to int (upper 128 bits)
                __m512i vb1i = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(vb, 1));
                // Shift left 16 bits and convert to float
                __m512 vb1 = _mm512_castsi512_ps(_mm512_slli_epi32(vb1i, 16));

                for (int mi = 0; mi < RM; ++mi) {
                    __m512 va0 = _mm512_loadu_ps(params.af + params.lda * (ii + mi) + ao);
                    __m512 va1 = _mm512_loadu_ps(params.af + params.lda * (ii + mi) + ao + 16);

                    // Multiply and accumulate
                    sums[mi][ni] = _mm512_fmadd_ps(va0, vb0, sums[mi][ni]);
                    sums[mi][ni] = _mm512_fmadd_ps(va1, vb1, sums[mi][ni]);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                float r = _mm512_reduce_add_ps(sums[mi][ni]);
                if (params.rs != NULL)
                    params.rs[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = fp32_to_bf16(r);
                else
                    params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = r;
            }
        }
    }
#else
    gemm_f32_bf16_256(m0, m, n0, n, RM, RN, params);
#endif
}
#endif //!ARM_NEON

void gemm_f32_bf16(int flags, const float *a, int aoffset, const short *b, int boffset, short *rs, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
{
    struct gemm_params p = {
                        .flags = flags,
                        .af = a,
                        .aoffset = aoffset,
                        .bs = b,
                        .boffset = boffset,
                        .rs = rs,
                        .r = r,
                        .roffset = roffset,
                        .m = m,
                        .n = n,
                        .k = k,
                        .ldaf = 0,
                        .ldbf = 0,
                        .lda = lda,
                        .ldb = ldb,
                        .ldc = ldc
    };

#if !defined(__ARM_NEON__)
    ((flags & HAS_AVX2) != 0)
           ? gemm(0, m, n0, n0 + n, gemm_f32_bf16_512, p)
           : gemm(0, m, n0, n0 + n, gemm_f32_bf16_256, p);
#else
    gemm(0, m, n0, n0 + n, gemm_f32_bf16_128_arm, p);
#endif
}

void gemm_f32_bf16_batch(int flags, int batch_num, const float *a, int aoffset, const short **b, int boffset, short **rs, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
{
    for (int i = 0; i < batch_num; i++) {
        gemm_f32_bf16(flags, a, aoffset, b[i], boffset, rs != NULL ? rs[i] : NULL, r != NULL ? r[i] : NULL, roffset, m, n0, n, k, lda, ldb, ldc);
    }
}
