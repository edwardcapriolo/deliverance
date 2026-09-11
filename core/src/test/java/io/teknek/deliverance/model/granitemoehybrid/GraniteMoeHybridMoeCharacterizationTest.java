package io.teknek.deliverance.model.granitemoehybrid;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
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

    @Test
    void reusedExpertScratchMatchesPerExpertAllocationAndPrintsTiming() {
        int tokens = 64;
        int experts = 8;
        int expertsPerToken = 2;
        int embedding = 256;
        int hidden = 768;
        int repeats = 2;
        FloatBufferTensor input = new FloatBufferTensor(tokens, embedding);
        FloatBufferTensor[] inputWeights = new FloatBufferTensor[experts];
        FloatBufferTensor[] outputWeights = new FloatBufferTensor[experts];
        for (int expert = 0; expert < experts; expert++) {
            inputWeights[expert] = new FloatBufferTensor(hidden * 2, embedding);
            outputWeights[expert] = new FloatBufferTensor(embedding, hidden);
            fill(inputWeights[expert]);
            fill(outputWeights[expert]);
        }
        fill(input);
        int[] selectedExperts = selectedExperts(tokens, expertsPerToken, experts);
        float[] selectedWeights = selectedWeights(tokens, expertsPerToken);
        int[] expertCounts = expertCounts(tokens, expertsPerToken, experts, selectedExperts);

        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            TensorAllocator allocator = new ArrayQueueTensorAllocator(new MetricRegistry());
            PanamaTensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool);
            FloatBufferTensor oldOutput = new FloatBufferTensor(tokens, embedding);
            FloatBufferTensor newOutput = new FloatBufferTensor(tokens, embedding);

            long oldStart = System.nanoTime();
            for (int i = 0; i < repeats; i++) {
                oldOutput.clear();
                runMoeWithPerExpertAllocation(ops, input, inputWeights, outputWeights, oldOutput, selectedExperts,
                        selectedWeights, expertCounts, experts, expertsPerToken, embedding, hidden);
            }
            long oldElapsedNanos = System.nanoTime() - oldStart;

            long newStart = System.nanoTime();
            for (int i = 0; i < repeats; i++) {
                newOutput.clear();
                runMoeWithReusedScratch(ops, input, inputWeights, outputWeights, newOutput, selectedExperts,
                        selectedWeights, expertCounts, experts, expertsPerToken, embedding, hidden);
            }
            long newElapsedNanos = System.nanoTime() - newStart;

            System.out.printf(java.util.Locale.ROOT,
                    "Granite MoE expert scratch characterization: old=%.3fms reused=%.3fms tokens=%d experts=%d embedding=%d hidden=%d repeats=%d%n",
                    oldElapsedNanos / 1_000_000.0, newElapsedNanos / 1_000_000.0, tokens, experts, embedding,
                    hidden, repeats);
            assertTensorEquals(oldOutput, newOutput, 1.0e-3f);
        }
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

    private static void runMoeWithPerExpertAllocation(PanamaTensorOperations ops, FloatBufferTensor input,
            FloatBufferTensor[] inputWeights, FloatBufferTensor[] outputWeights, FloatBufferTensor output,
            int[] selectedExperts, float[] selectedWeights, int[] expertCounts, int experts, int expertsPerToken,
            int embedding, int hiddenLength) {
        for (int expert = 0; expert < experts; expert++) {
            int tokenCount = expertCounts[expert];
            if (tokenCount == 0) {
                continue;
            }
            FloatBufferTensor expertInput = new FloatBufferTensor(tokenCount, embedding);
            FloatBufferTensor inputProjection = new FloatBufferTensor(tokenCount, hiddenLength * 2);
            FloatBufferTensor hidden = new FloatBufferTensor(tokenCount, hiddenLength);
            FloatBufferTensor down = new FloatBufferTensor(tokenCount, embedding);
            int[] sourceRows = new int[tokenCount];
            float[] routeWeights = new float[tokenCount];
            copySelectedRows(input, selectedExperts, selectedWeights, expert, expertsPerToken, embedding, expertInput,
                    sourceRows, routeWeights);
            ops.batchDotProduct(inputProjection, expertInput, inputWeights[expert], 0, 0, embedding, 0, 0,
                    hiddenLength * 2);
            scalarActivationGate(inputProjection, hidden, hiddenLength);
            ops.batchDotProduct(down, hidden, outputWeights[expert], 0, 0, hiddenLength, 0, 0, embedding);
            scalarScatterAdd(output, down, sourceRows, routeWeights, embedding);
        }
    }

    private static void runMoeWithReusedScratch(PanamaTensorOperations ops, FloatBufferTensor input,
            FloatBufferTensor[] inputWeights, FloatBufferTensor[] outputWeights, FloatBufferTensor output,
            int[] selectedExperts, float[] selectedWeights, int[] expertCounts, int experts, int expertsPerToken,
            int embedding, int hiddenLength) {
        int maxTokenCount = 0;
        for (int count : expertCounts) {
            maxTokenCount = Math.max(maxTokenCount, count);
        }
        FloatBufferTensor expertInputScratch = new FloatBufferTensor(maxTokenCount, embedding);
        FloatBufferTensor inputProjectionScratch = new FloatBufferTensor(maxTokenCount, hiddenLength * 2);
        FloatBufferTensor hiddenScratch = new FloatBufferTensor(maxTokenCount, hiddenLength);
        FloatBufferTensor downScratch = new FloatBufferTensor(maxTokenCount, embedding);
        int[] sourceRows = new int[maxTokenCount];
        float[] routeWeights = new float[maxTokenCount];
        for (int expert = 0; expert < experts; expert++) {
            int tokenCount = expertCounts[expert];
            if (tokenCount == 0) {
                continue;
            }
            try (AbstractTensor expertInput = expertInputScratch.firstRowsView(tokenCount);
                 AbstractTensor inputProjection = inputProjectionScratch.firstRowsView(tokenCount);
                 AbstractTensor hidden = hiddenScratch.firstRowsView(tokenCount);
                 AbstractTensor down = downScratch.firstRowsView(tokenCount)) {
                copySelectedRows(input, selectedExperts, selectedWeights, expert, expertsPerToken, embedding,
                        expertInput, sourceRows, routeWeights);
                ops.batchDotProduct(inputProjection, expertInput, inputWeights[expert], 0, 0, embedding, 0, 0,
                        hiddenLength * 2);
                scalarActivationGate(inputProjection, hidden, hiddenLength);
                ops.batchDotProduct(down, hidden, outputWeights[expert], 0, 0, hiddenLength, 0, 0, embedding);
                scalarScatterAdd(output, down, sourceRows, routeWeights, embedding, tokenCount);
            }
        }
    }

    private static void copySelectedRows(FloatBufferTensor input, int[] selectedExperts, float[] selectedWeights,
            int expert, int expertsPerToken, int embedding, AbstractTensor expertInput, int[] sourceRows,
            float[] routeWeights) {
        int outRow = 0;
        for (int token = 0; token < input.shape().first(); token++) {
            int offset = token * expertsPerToken;
            for (int top = 0; top < expertsPerToken; top++) {
                int selectedIndex = offset + top;
                if (selectedExperts[selectedIndex] == expert) {
                    expertInput.copyFrom(input, input.getOffset(token, 0), expertInput.getOffset(outRow, 0), embedding);
                    sourceRows[outRow] = token;
                    routeWeights[outRow] = selectedWeights[selectedIndex];
                    outRow++;
                }
            }
        }
    }

    private static int[] selectedExperts(int tokens, int expertsPerToken, int experts) {
        int[] selectedExperts = new int[tokens * expertsPerToken];
        for (int token = 0; token < tokens; token++) {
            for (int top = 0; top < expertsPerToken; top++) {
                selectedExperts[token * expertsPerToken + top] = (token * 3 + top * 5) % experts;
            }
        }
        return selectedExperts;
    }

    private static float[] selectedWeights(int tokens, int expertsPerToken) {
        float[] selectedWeights = new float[tokens * expertsPerToken];
        for (int token = 0; token < tokens; token++) {
            selectedWeights[token * expertsPerToken] = 0.65f;
            selectedWeights[token * expertsPerToken + 1] = 0.35f;
        }
        return selectedWeights;
    }

    private static int[] expertCounts(int tokens, int expertsPerToken, int experts, int[] selectedExperts) {
        int[] counts = new int[experts];
        for (int token = 0; token < tokens; token++) {
            int offset = token * expertsPerToken;
            for (int top = 0; top < expertsPerToken; top++) {
                counts[selectedExperts[offset + top]]++;
            }
        }
        return counts;
    }

    private static void scalarActivationGate(AbstractTensor inputProjection, AbstractTensor hidden, int hiddenLength) {
        for (int row = 0; row < inputProjection.shape().first(); row++) {
            for (int col = 0; col < hiddenLength; col++) {
                float gate = inputProjection.get(row, col);
                float up = inputProjection.get(row, hiddenLength + col);
                hidden.set(ActivationFunction.eval(ActivationFunction.Type.SILU, gate) * up, row, col);
            }
        }
    }

    private static void scalarScatterAdd(AbstractTensor output, AbstractTensor down, int[] sourceRows,
            float[] routeWeights, int hidden, int tokenCount) {
        for (int row = 0; row < tokenCount; row++) {
            int outputRow = sourceRows[row];
            float weight = routeWeights[row];
            for (int col = 0; col < hidden; col++) {
                output.set(output.get(outputRow, col) + down.get(row, col) * weight, outputRow, col);
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
