#ifndef TENSOR2_NATIVE_H
#define TENSOR2_NATIVE_H

typedef enum tensor2_status {
    TENSOR2_OK = 0,
    TENSOR2_UNSUPPORTED = 1
} tensor2_status;

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
        int b_stride);

#endif
