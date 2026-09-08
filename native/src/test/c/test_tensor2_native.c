#include "tensor2_native.h"

int main(void) {
    float a[1] = {1.0f};
    float b[1] = {2.0f};
    return tensor2_maccumulate_f32_f32(a, b, 1, 1, 0, 1, 1, 1, 0) == TENSOR2_UNSUPPORTED ? 0 : 1;
}
