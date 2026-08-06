#ifndef GPU_DOT_H
#define GPU_DOT_H

#include <stdint.h>

//Returns the memory free on the GPU and the max group size
void init_gpu(int64_t *results);

//Returns a unique identifier for the tensor
int64_t register_tensor(const char *data, int size);

void unregister_tensor(int64_t id);

int64_t register_scratch_buffers(int params_size, int input_size, int result_size);

//Returns a unique identifier for the shader
int64_t register_shader(const char *data, int size);

//GEMM F32/BF16/Q4
void gpu_gemm(int64_t scratch_id, int64_t shader, const void *a, const void *a2, int aoffset, int alimit, int64_t bid, int64_t bid2, int boffset, int blimit, float *r, int roffset, int rlimit, int m, int n0, int n, int k, int lda, int ldb, int ldc, int m1_optimized);
void gpu_gemm_batch(int64_t scratch_id, int64_t shader, int batch_num, const void *a, const void *a2, int aoffset, int alimit, const int64_t *bid, const int64_t *bid2, int boffset, int blimit, float **r, int roffset, int rlimit, int m, int n0, int n, int k, int lda, int ldb, int ldc, int m1_optimized);
void gpu_decode_paged_attention_head(int64_t scratch_id, int64_t shader, const void *query, int qoffset, const int64_t *kid, const int64_t *vid, int page_count, int visible_rows, int page_rows, int head_size, int kv_offset, int key_stride, int value_stride, float *out, int out_offset, int out_stride, float scale, int key_buffer_size, int value_buffer_size);
void gpu_decode_attention_packed_head(int64_t scratch_id, int64_t shader, const void *query, int qoffset, const void *key, int key_size, const void *value, int value_size, float *out, int out_offset, int visible_rows, int head_size, float scale);
void gpu_decode_attention_packed_all_heads(int64_t scratch_id, int64_t shader, const void *query, int qoffset, int query_size, const void *key, int key_size, const void *value, int value_size, float *out, int out_offset, int visible_rows, int number_of_heads, int number_of_kv_heads, int head_size, int kv_length, float scale);

#endif
