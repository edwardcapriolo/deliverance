package io.teknek.deliverance.model.diffusiongemma;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LinearTemperatureScheduleLogitsProcessorTest {

    @Test
    void testLinearTemperatureScheduleFinalStep() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2));
             FloatBufferTensor scores = ones(1, 10)) {
            LinearTemperatureScheduleLogitsProcessor processor = processor(pool, 0.4f, 0.8f, 48);

            processor.process(scores, 48);

            assertAll(scores, 1.0f / 0.8f);
        }
    }

    @Test
    void testLinearTemperatureScheduleMidpoint() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2));
             FloatBufferTensor scores = ones(1, 10)) {
            LinearTemperatureScheduleLogitsProcessor processor = processor(pool, 0.4f, 0.8f, 48);

            processor.process(scores, 24);

            assertAll(scores, 1.0f / 0.6f);
        }
    }

    @Test
    void rejectsBadScheduleParameters() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2))) {
            assertThrows(IllegalArgumentException.class, () -> processor(pool, -0.1f, 0.8f, 48));
            assertThrows(IllegalArgumentException.class, () -> processor(pool, 0.8f, 0.4f, 48));
            assertThrows(IllegalArgumentException.class, () -> processor(pool, 0.4f, 0.8f, 0));
        }
    }

    @Test
    void rejectsOutOfRangeCurrentStep() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2));
             FloatBufferTensor scores = ones(1, 10)) {
            LinearTemperatureScheduleLogitsProcessor processor = processor(pool, 0.4f, 0.8f, 48);

            assertThrows(IllegalArgumentException.class, () -> processor.process(scores, -1));
            assertThrows(IllegalArgumentException.class, () -> processor.process(scores, 49));
        }
    }

    private static LinearTemperatureScheduleLogitsProcessor processor(WrappedForkJoinPool pool, float tMin, float tMax,
            int maxDenoisingSteps) {
        MetricRegistry metrics = new MetricRegistry();
        return new LinearTemperatureScheduleLogitsProcessor(tMin, tMax, maxDenoisingSteps,
                new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, new ArrayQueueTensorAllocator(metrics), pool),
                metrics);
    }

    private static FloatBufferTensor ones(int rows, int cols) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(1.0f, row, col);
            }
        }
        return tensor;
    }

    private static void assertAll(FloatBufferTensor tensor, float expected) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int col = 0; col < tensor.shape().last(); col++) {
                assertEquals(expected, tensor.get(row, col), 1.0e-6f, "row=" + row + " col=" + col);
            }
        }
    }
}
