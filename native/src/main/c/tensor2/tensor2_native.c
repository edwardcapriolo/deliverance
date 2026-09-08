#include "tensor2_native.h"

tensor2_status tensor2_maccumulate_f32_f32(
        float *a,
        const float *b,
        int rows,
        int columns,
        int offset,
        int length,
        int a_stride,
        int b_stride,
        int b_broadcast) {
    (void) a;
    (void) b;
    (void) rows;
    (void) columns;
    (void) offset;
    (void) length;
    (void) a_stride;
    (void) b_stride;
    (void) b_broadcast;
    return TENSOR2_UNSUPPORTED;
}
