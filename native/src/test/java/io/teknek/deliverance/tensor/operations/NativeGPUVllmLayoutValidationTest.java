package io.teknek.deliverance.tensor.operations;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class NativeGPUVllmLayoutValidationTest {

    @Test
    void acceptsValidQwenLikeVllmLayout() {
        try (AbstractTensor out = new FloatBufferTensor(1, 4096);
             AbstractTensor query = new FloatBufferTensor(1, 4096);
             AbstractTensor keyCache = new FloatBufferTensor(8 * 32, 1024);
             AbstractTensor valueCache = new FloatBufferTensor(8 * 32, 1024)) {
            assertDoesNotThrow(() -> GpuFlashDecodeShape.validateVllmLayout(out, query, keyCache,
                    valueCache, new int[] { 2, 5, 0, 3, 6, 1, 4, 7 }, 256, 32, 32, 8, 128));
        }
    }

    @Test
    void rejectsBlockTableOutsideCache() {
        try (AbstractTensor out = new FloatBufferTensor(1, 4096);
             AbstractTensor query = new FloatBufferTensor(1, 4096);
            AbstractTensor keyCache = new FloatBufferTensor(8 * 32, 1024);
             AbstractTensor valueCache = new FloatBufferTensor(8 * 32, 1024)) {
            assertThrows(IllegalArgumentException.class,
                    () -> GpuFlashDecodeShape.validateVllmLayout(out, query, keyCache, valueCache,
                            new int[] { 8 }, 32, 32, 32, 8, 128));
        }
    }

    @Test
    void rejectsUnsupportedHeadSizeForCurrentWorkgroupShape() {
        try (AbstractTensor out = new FloatBufferTensor(1, 4096);
             AbstractTensor query = new FloatBufferTensor(1, 4096);
            AbstractTensor keyCache = new FloatBufferTensor(8 * 32, 2048);
             AbstractTensor valueCache = new FloatBufferTensor(8 * 32, 2048)) {
            assertThrows(IllegalArgumentException.class,
                    () -> GpuFlashDecodeShape.validateVllmLayout(out, query, keyCache, valueCache,
                            new int[] { 0 }, 32, 32, 16, 8, 256));
        }
    }
}
