#include "../../main/c/gpu/vector_gpu.h"

#include <stdint.h>

typedef void (*gpu_gemm_signature)(
    int64_t scratch_id,
    int64_t shader,
    const void *a,
    const void *a2,
    int aoffset,
    int alimit,
    int64_t bid,
    int64_t bid2,
    int boffset,
    int blimit,
    float *r,
    int roffset,
    int rlimit,
    int m,
    int n0,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int m1_optimized
);

typedef void (*gpu_gemm_batch_signature)(
    int64_t scratch_id,
    int64_t shader,
    int batch_num,
    const void *a,
    const void *a2,
    int aoffset,
    int alimit,
    const int64_t *bid,
    const int64_t *bid2,
    int boffset,
    int blimit,
    float **r,
    int roffset,
    int rlimit,
    int m,
    int n0,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int m1_optimized
);

int main(void) {
    gpu_gemm_signature single = gpu_gemm;
    gpu_gemm_batch_signature batch = gpu_gemm_batch;
    return single == 0 || batch == 0;
}
