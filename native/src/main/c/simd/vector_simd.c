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
                int aoo = ao;
                int boo = bo;

                for (int mi = 0; mi < RM; ++mi) {
                    ao = aoo;
                    bo = boo;

                    // Load float32
                    float32x4_t ablock = vld1q_f32(params.af + (params.ldaf * (ii + mi) + (ao / Q4_BLOCK_SIZE)));
                    float32x4_t bblock = vld1q_f32(params.bf + (params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)));
                    float32x4_t scaled = vmulq_f32(ablock, bblock);
                    vst1q_f32(scalef, scaled);

                    for(int j = 0; j < 4; j++, ao += 32, bo += 16) {
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

void gemm_f32_q4(int flags, const float *a, int aoffset, const float *bf, const char* b, int boffset, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc)
{
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
            for (int i = 0; i < params.k / Q4_BLOCK_SIZE; i += 4) {
                int aoo = ao;
                int boo = bo;
                for (int mi = 0; mi < RM; ++mi) {
                    ao = aoo;
                    bo = boo;
                    float32x4_t bblock = vld1q_f32(params.bf + (params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)));
                    vst1q_f32(scalef, bblock);
                    for(int j = 0; j < 4; j++, ao += 32, bo += 16) {
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

#if !defined(__ARM_NEON__)
  gemm()
#else
#endif

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
