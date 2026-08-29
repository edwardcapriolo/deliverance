package io.teknek.deliverance.tensor;

import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TrackedReadOnlyTensorTest {

    @Test
    void directWritesThrowImmediately() {
        try (AbstractTensor backing = tensor(1.0f);
             AbstractTensor source = tensor(10.0f);
             TrackedReadOnlyTensor tracked = new TrackedReadOnlyTensor(backing)) {
            assertThrows(IllegalStateException.class, () -> tracked.set(99.0f, 0, 0));
            assertThrows(IllegalStateException.class, () -> tracked.copyFrom(source, 0, 0, 4));
            assertThrows(IllegalStateException.class, tracked::clear);
            assertFalse(tracked.hasChecksumChanged());
        }
    }

    @Test
    void delegateMutationChangesChecksumAndCloseThrows() {
        try (AbstractTensor backing = tensor(1.0f)) {
            TrackedReadOnlyTensor tracked = new TrackedReadOnlyTensor(backing);

            assertFalse(tracked.hasChecksumChanged());
            backing.set(99.0f, 0, 0);

            assertTrue(tracked.hasChecksumChanged());
            assertThrows(IllegalStateException.class, tracked::close);
        }
    }

    @Test
    void memorySegmentMutationChangesChecksumAndCloseThrows() {
        try (AbstractTensor backing = tensor(1.0f)) {
            TrackedReadOnlyTensor tracked = new TrackedReadOnlyTensor(backing);

            assertFalse(tracked.hasChecksumChanged());
            tracked.getMemorySegment().setAtIndex(java.lang.foreign.ValueLayout.JAVA_BYTE, 0, (byte) 1);

            assertTrue(tracked.hasChecksumChanged());
            assertThrows(IllegalStateException.class, tracked::close);
        }
    }

    private static AbstractTensor tensor(float firstValue) {
        AbstractTensor tensor = new FloatBufferTensor(TensorShape.of(1, 4));
        for (int i = 0; i < 4; i++) {
            tensor.set(firstValue + i, 0, i);
        }
        return tensor;
    }
}
