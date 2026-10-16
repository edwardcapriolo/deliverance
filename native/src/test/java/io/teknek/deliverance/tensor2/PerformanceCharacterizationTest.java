package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Measures tensor2 projections at dimensions used by the Qwen3 0.6B and 4B models.
 *
 * <p>This is deliberately a performance characterization rather than a correctness test. The
 * fuzzer covers correctness and edge shapes; this test keeps the logical model dimensions while
 * using deterministic synthetic tensors so the kernel strategies can be compared without a model
 * download.</p>
 */
class PerformanceCharacterizationTest {
    private static final int WARMUPS = 2;
    private static final int REPETITIONS = 8;

    @ParameterizedTest(name = "{0} {1}")
    @MethodSource("modelCases")
    void modelSizedProjection(ModelCase model, OperandMode mode) {
        GemmKernel kernel = requestedKernel();
        Threading threading = requestedThreading();
        Assumptions.assumeTrue(mode != OperandMode.F32_F32 || kernel != GemmKernel.CHUNK,
                "CHUNK is currently implemented only for Q4 projections");

        MetricRegistry metrics = new MetricRegistry();
        TensorProviderKind provider = requestedProvider();
        NativeOps nativeOps = provider == TensorProviderKind.SIMD ? new NativeOps() : null;
        try {
            TensorOps operations = provider == TensorProviderKind.SIMD ? nativeOps : new PanamaOps();
            Lighter lighter = new Lighter(metrics, Map.of(provider, operations));
            if (mode == OperandMode.F32_F32) {
                try (TensorRef input = lighter.allocate(DType.F32,
                             TensorShape.of(model.rows(), model.inputColumns()));
                     TensorRef weights = lighter.allocate(DType.F32,
                             TensorShape.of(model.outputColumns(), model.inputColumns()));
                     TensorRef output = lighter.allocate(DType.F32,
                             TensorShape.of(model.rows(), model.outputColumns()))) {
                    fill(input, model.seed());
                    fill(weights, model.seed() + 17);
                    characterize(lighter, output, input, weights, model, mode, provider, kernel, threading);
                }
                return;
            }

            try (TensorRef denseInput = lighter.allocate(DType.F32,
                         TensorShape.of(model.rows(), model.inputColumns()));
                 TensorRef denseWeights = lighter.allocate(DType.F32,
                         TensorShape.of(model.outputColumns(), model.inputColumns()))) {
                fill(denseInput, model.seed());
                fill(denseWeights, model.seed() + 17);
                Lighter quantizer = new Lighter(new MetricRegistry(),
                        Map.of(TensorProviderKind.PANAMA, new PanamaOps()));
                try (TensorRef input = quantizer.reshape(denseInput, DType.I8);
                     TensorRef weights = quantizer.reshape(denseWeights, DType.Q4);
                     TensorRef output = lighter.allocate(DType.F32,
                             TensorShape.of(model.rows(), model.outputColumns()))) {
                    characterize(lighter, output, input, weights, model, mode, provider, kernel, threading);
                }
            }
        } finally {
            if (nativeOps != null) nativeOps.close();
        }
    }

    private static void characterize(Lighter lighter, TensorRef output, TensorRef input, TensorRef weights,
            ModelCase model, OperandMode mode, TensorProviderKind provider, GemmKernel kernel, Threading threading) {
        long warmupStart = System.nanoTime();
        for (int i = 0; i < WARMUPS; i++) {
            execute(lighter, output, input, weights, model, kernel, threading);
        }
        long warmupNanos = System.nanoTime() - warmupStart;

        long start = System.nanoTime();
        for (int i = 0; i < REPETITIONS; i++) {
            execute(lighter, output, input, weights, model, kernel, threading);
        }
        long elapsedNanos = System.nanoTime() - start;
        System.out.printf(java.util.Locale.ROOT,
                "[tensor2-performance] provider=%s model=%s projection=%s mode=%s rows=%d input=%d output=%d "
                        + "kernel=%s threading=%s warmup_ms=%.3f mean_ms=%.3f%n",
                provider, model.model(), model.projection(), mode, model.rows(), model.inputColumns(), model.outputColumns(),
                kernel, threading, warmupNanos / 1_000_000.0 / WARMUPS,
                elapsedNanos / 1_000_000.0 / REPETITIONS);
    }

    private static void execute(Lighter lighter, TensorRef output, TensorRef input, TensorRef weights,
            ModelCase model, GemmKernel kernel, Threading threading) {
        if (kernel == GemmKernel.BATCH) {
            lighter.batchDotProduct(new BatchDotProduct()
                    .result(output).a(input).b(weights)
                    .aColumnOffset(0).bColumnOffset(0).columnLength(model.inputColumns())
                    .resultRowOffset(0).bRowOffset(0).rowChunkSize(model.outputColumns())
                    .kernel(kernel).threading(threading));
        } else {
            lighter.dotProductRows(output, input, weights, 0, 0, model.inputColumns(),
                    0, model.outputColumns(), 0);
        }
    }

    private static void fill(TensorRef tensor, int seed) {
        Random random = new Random(seed);
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.set(random.nextFloat() - 0.5f, row, column);
            }
        }
    }

    static Stream<Arguments> modelCases() {
        return Stream.of(
                model("qwen06b", "mlp_gate_up", 1024, 3072, 11),
                model("qwen06b", "mlp_down", 3072, 1024, 13),
                model("qwen06b", "query", 1024, 2048, 17),
                model("qwen06b", "key_value", 1024, 1024, 19),
                model("qwen4b", "mlp_gate_up", 2560, 9728, 23),
                model("qwen4b", "mlp_down", 9728, 2560, 29),
                model("qwen4b", "query", 2560, 4096, 31),
                model("qwen4b", "key_value", 2560, 1024, 37))
                .flatMap(model -> Stream.of(OperandMode.values())
                        .map(mode -> Arguments.of(model, mode)));
    }

    private static ModelCase model(String model, String projection, int inputColumns, int outputColumns,
            int seed) {
        return new ModelCase(model, projection, 1, inputColumns, outputColumns, seed);
    }

    private static GemmKernel requestedKernel() {
        return GemmKernel.ROW;
    }

    private static Threading requestedThreading() {
        return Threading.PROVIDER;
    }

    private static TensorProviderKind requestedProvider() {
        return TensorProviderKind.PANAMA;
    }

    private enum OperandMode {
        F32_F32,
        I8_Q4
    }

    private record ModelCase(String model, String projection, int rows, int inputColumns, int outputColumns,
            int seed) {
        @Override
        public String toString() {
            return model + " " + projection;
        }
    }
}
