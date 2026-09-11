package io.teknek.deliverance.model.granitemoehybrid;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GraniteMoeHybridMoeCharacterizationTest {

    @Test
    void sharedVectorActivationGateMatchesMoeScalarGateWithBetterShape() {
        int rows = 8;
        int hidden = 256;
        FloatBufferTensor projected = new FloatBufferTensor(rows, hidden * 2);
        fill(projected);
        FloatBufferTensor oldHidden = new FloatBufferTensor(rows, hidden);
        FloatBufferTensor newHidden = new FloatBufferTensor(rows, hidden);

        long oldStart = System.nanoTime();
        scalarActivationGate(projected, oldHidden, hidden);
        long oldElapsedNanos = System.nanoTime() - oldStart;

        long newStart = System.nanoTime();
        GraniteMoeHybridSharedMlp.applyActivationGate(projected, newHidden, ActivationFunction.Type.SILU, hidden);
        long newElapsedNanos = System.nanoTime() - newStart;

        System.out.printf(java.util.Locale.ROOT,
                "Granite MoE activation gate characterization: scalar=%.3fms vector=%.3fms rows=%d hidden=%d%n",
                oldElapsedNanos / 1_000_000.0, newElapsedNanos / 1_000_000.0, rows, hidden);
        assertTensorEquals(oldHidden, newHidden, 1.0e-4f);
    }

    @Test
    void providerSaxpyScatterAddMatchesMoeScalarScatterAdd() {
        int routedRows = 32;
        int outputRows = 8;
        int hidden = 512;
        int[] sourceRows = new int[routedRows];
        float[] routeWeights = new float[routedRows];
        FloatBufferTensor down = new FloatBufferTensor(routedRows, hidden);
        FloatBufferTensor oldOutput = new FloatBufferTensor(outputRows, hidden);
        FloatBufferTensor newOutput = new FloatBufferTensor(outputRows, hidden);
        fill(down);
        for (int row = 0; row < routedRows; row++) {
            sourceRows[row] = row % outputRows;
            routeWeights[row] = 0.1f + (row % 5) * 0.05f;
        }

        long oldStart = System.nanoTime();
        scalarScatterAdd(oldOutput, down, sourceRows, routeWeights, hidden);
        long oldElapsedNanos = System.nanoTime() - oldStart;

        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            TensorAllocator allocator = new ArrayQueueTensorAllocator(new MetricRegistry());
            PanamaTensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool);
            long newStart = System.nanoTime();
            saxpyScatterAdd(ops, newOutput, down, sourceRows, routeWeights, hidden);
            long newElapsedNanos = System.nanoTime() - newStart;

            System.out.printf(java.util.Locale.ROOT,
                    "Granite MoE scatter-add characterization: scalar=%.3fms saxpy=%.3fms routed_rows=%d hidden=%d%n",
                    oldElapsedNanos / 1_000_000.0, newElapsedNanos / 1_000_000.0, routedRows, hidden);
        }
        assertTensorEquals(oldOutput, newOutput, 1.0e-4f);
    }

    private static void scalarActivationGate(FloatBufferTensor inputProjection, FloatBufferTensor hidden, int hiddenLength) {
        for (int row = 0; row < inputProjection.shape().first(); row++) {
            for (int col = 0; col < hiddenLength; col++) {
                float gate = inputProjection.get(row, col);
                float up = inputProjection.get(row, hiddenLength + col);
                hidden.set(ActivationFunction.eval(ActivationFunction.Type.SILU, gate) * up, row, col);
            }
        }
    }

    private static void scalarScatterAdd(FloatBufferTensor output, FloatBufferTensor down, int[] sourceRows,
            float[] routeWeights, int hidden) {
        for (int row = 0; row < sourceRows.length; row++) {
            int outputRow = sourceRows[row];
            float weight = routeWeights[row];
            for (int col = 0; col < hidden; col++) {
                output.set(output.get(outputRow, col) + down.get(row, col) * weight, outputRow, col);
            }
        }
    }

    private static void saxpyScatterAdd(PanamaTensorOperations ops, FloatBufferTensor output, FloatBufferTensor down,
            int[] sourceRows, float[] routeWeights, int hidden) {
        for (int row = 0; row < sourceRows.length; row++) {
            try (var downRow = down.slice(row); var outputRow = output.slice(sourceRows[row])) {
                ops.saxpy(routeWeights[row], downRow, outputRow, 0, 0, hidden);
            }
        }
    }

    private static void fill(FloatBufferTensor tensor) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.set(((row + 1) * (column % 17 - 8)) / 16.0f, row, column);
            }
        }
    }

    private static void assertTensorEquals(FloatBufferTensor expected, FloatBufferTensor actual, float tolerance) {
        assertEquals(expected.shape(), actual.shape());
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int column = 0; column < expected.shape().last(); column++) {
                assertEquals(expected.get(row, column), actual.get(row, column), tolerance,
                        "row=" + row + " column=" + column);
            }
        }
    }
}
