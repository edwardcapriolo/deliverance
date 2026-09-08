#ifndef TENSOR2_NATIVE_H
#define TENSOR2_NATIVE_H

typedef enum tensor2_status {
    TENSOR2_OK = 0,
    TENSOR2_UNSUPPORTED = 1
} tensor2_status;

tensor2_status tensor2_maccumulate_f32_f32(
        float *a,
        const float *b,
        int rows,
        int columns,
        int offset,
        int length,
        int a_stride,
        int b_stride,
        int b_broadcast);

#endif
