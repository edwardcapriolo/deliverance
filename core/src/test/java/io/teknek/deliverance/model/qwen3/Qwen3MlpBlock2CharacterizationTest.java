package io.teknek.deliverance.model.qwen3;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.generator2.MLPBlock2;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.NativeOps;
import io.teknek.deliverance.tensor2.TensorProviderKind;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ForkJoinPool;

class Qwen3MlpBlock2CharacterizationTest {
    @Test
    void qwen3FourBDecodeMlpBlock2MatchesLegacyAndReportsTiming() {
        Assumptions.assumeTrue(NativeOps.isAvailable(), "native Tensor2 unavailable");
        int embedding = 2560;
        int hidden = 9728;
        MetricRegistry metrics = new MetricRegistry();
        ArrayQueueTensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        Lighter lighter = new Lighter(metrics);
        lighter.putTensorOperations(TensorProviderKind.SIMD, new NativeOps());
        Config config = new Config(4096, embedding, hidden, 32, 8, 1, 1.0e-6f, 32_000, 0,
                List.of(1), ActivationFunction.Type.SILU, 1_000_000.0, null, 128, null, null);
        NativeSimdTensorOperations legacyOps = new NativeSimdTensorOperations(
                new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator,
                        new WrappedForkJoinPool(new ForkJoinPool(16))), 64);
        AbstractModel model = Mockito.mock(AbstractModel.class);
        Mockito.when(model.getConfig()).thenReturn(config);
        Mockito.when(model.getMetricRegistry()).thenReturn(metrics);
        Mockito.when(model.getWorkingQType()).thenReturn(DType.I8);
        Mockito.when(model.getTensorParallelContext()).thenReturn(new io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext(0, 1));
        Mockito.when(model.getCompositeOps()).thenReturn(new CompositeOps(lighter));
        Mockito.when(model.primaryTensorOperations()).thenReturn(legacyOps);
        Mockito.when(model.makeDenseTensorRef(Mockito.anyInt(), Mockito.anyInt()))
                .thenAnswer(invocation -> lighter.allocate(DType.F32, TensorShape.of(
                        invocation.getArgument(0), invocation.getArgument(1))));
        Mockito.doAnswer(invocation -> {
            io.teknek.deliverance.math.BiIntConsumer action = invocation.getArgument(5);
            action.accept(invocation.getArgument(1), invocation.getArgument(2));
            return null;
        }).when(model).runChunks(Mockito.anyString(), Mockito.anyInt(), Mockito.anyInt(), Mockito.anyInt(),
                Mockito.any(Optional.class), Mockito.any(io.teknek.deliverance.math.BiIntConsumer.class));
        Mockito.when(model.activeLoraDeltaFor(Mockito.anyString())).thenReturn(Optional.empty());

        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(16));
             FloatBufferTensor denseInput = tensor(1, embedding, 501);
             Q8ByteBufferTensor inputLegacy = new Q8ByteBufferTensor(denseInput);
             Q4ByteBufferTensor gateLegacy = new Q4ByteBufferTensor(tensor(hidden, embedding, 502));
             Q4ByteBufferTensor upLegacy = new Q4ByteBufferTensor(tensor(hidden, embedding, 503));
             Q4ByteBufferTensor downLegacy = new Q4ByteBufferTensor(tensor(embedding, hidden, 504));
             TensorRef input = TensorRef.borrowed(inputLegacy);
             TensorRef gateWeight = TensorRef.borrowed(gateLegacy);
             TensorRef upWeight = TensorRef.borrowed(upLegacy);
             TensorRef downWeight = TensorRef.borrowed(downLegacy)) {
            Mockito.when(model.getPool()).thenReturn(pool);
             MLPBlock2 block = new MLPBlock2(model, gateWeight, upWeight, downWeight, lighter);
             componentCharacterization(legacyOps, lighter, denseInput, inputLegacy, gateLegacy, upLegacy,
                     downLegacy, input, gateWeight, upWeight, downWeight, embedding, hidden, pool);
             for (int i = 0; i < 2; i++) {
                legacyMlp(legacyOps, inputLegacy, gateLegacy, upLegacy, downLegacy, embedding, hidden);
                try (TensorRef ignored = block.forward(input, Optional.empty(), ForwardPhase.DECODE)) {
                }
            }
            int repetitions = 4;
            long legacyStart = System.nanoTime();
            for (int i = 0; i < repetitions; i++) {
                try (AbstractTensor ignored = legacyMlp(legacyOps, inputLegacy, gateLegacy, upLegacy, downLegacy,
                        embedding, hidden)) {
                }
            }
            long legacyNanos = System.nanoTime() - legacyStart;
            long refStart = System.nanoTime();
            for (int i = 0; i < repetitions; i++) {
                try (TensorRef ignored = block.forward(input, Optional.empty(), ForwardPhase.DECODE)) {
                }
            }
            long refNanos = System.nanoTime() - refStart;
            System.out.printf("[qwen3-4b-mlp-block2] legacy_ms_per=%.3f tensorRef_ms_per=%.3f%n",
                    legacyNanos / 1_000_000.0 / repetitions, refNanos / 1_000_000.0 / repetitions);
        }
    }

    private static void componentCharacterization(NativeSimdTensorOperations legacyOps, Lighter lighter,
            FloatBufferTensor denseInput, Q8ByteBufferTensor inputLegacy, Q4ByteBufferTensor gateLegacy,
            Q4ByteBufferTensor upLegacy, Q4ByteBufferTensor downLegacy, TensorRef input, TensorRef gateWeight,
            TensorRef upWeight, TensorRef downWeight, int embedding, int hidden, WrappedForkJoinPool pool) {
        CompositeOps composite = new CompositeOps(lighter);
        int repetitions = 4;
        for (int i = 0; i < 2; i++) {
            try (FloatBufferTensor gate = new FloatBufferTensor(1, hidden);
                 FloatBufferTensor up = new FloatBufferTensor(1, hidden);
                 TensorRef gateRef = lighter.allocate(DType.F32, TensorShape.of(1, hidden));
                 TensorRef upRef = lighter.allocate(DType.F32, TensorShape.of(1, hidden))) {
                legacyOps.dotProductBatchChunk(new AbstractTensor[]{gate, up}, inputLegacy,
                        new AbstractTensor[]{gateLegacy, upLegacy}, 0, embedding, 0, hidden);
                lighter.dotProductBatchChunk(new io.teknek.deliverance.tensor2.DotProductBatchChunk()
                        .results(gateRef, upRef).input(input).weights(gateWeight, upWeight)
                        .inputColumnStart(0).weightColumnStart(0).columnLength(embedding)
                        .weightRowStart(0).weightRowCount(hidden).outputColumnStart(0));
            }
        }
        long legacyGateUpStart = System.nanoTime();
        for (int i = 0; i < repetitions; i++) {
            try (FloatBufferTensor gate = new FloatBufferTensor(1, hidden);
                 FloatBufferTensor up = new FloatBufferTensor(1, hidden)) {
                legacyOps.dotProductBatchChunk(new AbstractTensor[]{gate, up}, inputLegacy,
                        new AbstractTensor[]{gateLegacy, upLegacy}, 0, embedding, 0, hidden);
            }
        }
        long legacyGateUpNanos = System.nanoTime() - legacyGateUpStart;
        long refGateUpStart = System.nanoTime();
        for (int i = 0; i < repetitions; i++) {
            try (TensorRef gate = lighter.allocate(DType.F32, TensorShape.of(1, hidden));
                 TensorRef up = lighter.allocate(DType.F32, TensorShape.of(1, hidden))) {
                lighter.dotProductBatchChunk(new io.teknek.deliverance.tensor2.DotProductBatchChunk()
                        .results(gate, up).input(input).weights(gateWeight, upWeight)
                        .inputColumnStart(0).weightColumnStart(0).columnLength(embedding)
                        .weightRowStart(0).weightRowCount(hidden).outputColumnStart(0));
            }
        }
        long refGateUpNanos = System.nanoTime() - refGateUpStart;

        long legacyActivationStart = System.nanoTime();
        for (int i = 0; i < repetitions; i++) {
            try (FloatBufferTensor gate = new FloatBufferTensor(1, hidden);
                 FloatBufferTensor up = new FloatBufferTensor(1, hidden)) {
                legacyOps.dotProductBatchChunk(new AbstractTensor[]{gate, up}, inputLegacy,
                        new AbstractTensor[]{gateLegacy, upLegacy}, 0, embedding, 0, hidden);
                try (AbstractTensor ignored = legacyOps.activationMultiplyQuantize(gate, up,
                        ActivationFunction.Type.SILU, DType.I8, 0, hidden)) {
                }
            }
        }
        long legacyActivationNanos = System.nanoTime() - legacyActivationStart;
        long refActivationStart = System.nanoTime();
        for (int i = 0; i < repetitions; i++) {
            try (TensorRef gate = lighter.allocate(DType.F32, TensorShape.of(1, hidden));
                 TensorRef up = lighter.allocate(DType.F32, TensorShape.of(1, hidden))) {
                lighter.dotProductBatchChunk(new io.teknek.deliverance.tensor2.DotProductBatchChunk()
                        .results(gate, up).input(input).weights(gateWeight, upWeight)
                        .inputColumnStart(0).weightColumnStart(0).columnLength(embedding)
                        .weightRowStart(0).weightRowCount(hidden).outputColumnStart(0));
                try (TensorRef ignored = composite.activationMultiplyQuantize(
                        new io.teknek.deliverance.tensor2.ActivationMultiplyQuantize(gate, up,
                                ActivationFunction.Type.SILU, DType.I8).offsetAndLength(0, hidden))) {
                }
            }
        }
        long refActivationNanos = System.nanoTime() - refActivationStart;

        long legacyDownStart = System.nanoTime();
        for (int i = 0; i < repetitions; i++) {
            try (AbstractTensor gate = legacyOps.activationMultiplyQuantize(
                    new FloatBufferTensor(1, hidden), new FloatBufferTensor(1, hidden),
                    ActivationFunction.Type.SILU, DType.I8, 0, hidden);
                 AbstractTensor output = new FloatBufferTensor(1, embedding)) {
                legacyOps.dotProductChunk(output, gate, downLegacy, 0, hidden, 0, embedding);
            }
        }
        long legacyDownNanos = System.nanoTime() - legacyDownStart;
        long refDownStart = System.nanoTime();
        for (int i = 0; i < repetitions; i++) {
            try (TensorRef denseGate = lighter.allocate(DType.F32, TensorShape.of(1, hidden));
                 TensorRef denseUp = lighter.allocate(DType.F32, TensorShape.of(1, hidden));
                 TensorRef output = lighter.allocate(DType.F32, TensorShape.of(1, embedding))) {
                try (TensorRef gate = composite.activationMultiplyQuantize(
                        new io.teknek.deliverance.tensor2.ActivationMultiplyQuantize(denseGate, denseUp,
                                ActivationFunction.Type.SILU, DType.I8).offsetAndLength(0, hidden))) {
                    lighter.dotProductRows(output, gate, downWeight, 0, hidden, 0, embedding, 0);
                }
            }
        }
        long refDownNanos = System.nanoTime() - refDownStart;
        System.out.printf("[qwen3-4b-mlp-components] gate_up_legacy_ms=%.3f gate_up_ref_ms=%.3f "
                        + "activation_legacy_ms=%.3f activation_ref_ms=%.3f down_legacy_ms=%.3f down_ref_ms=%.3f%n",
                legacyGateUpNanos / 1_000_000.0 / repetitions, refGateUpNanos / 1_000_000.0 / repetitions,
                legacyActivationNanos / 1_000_000.0 / repetitions, refActivationNanos / 1_000_000.0 / repetitions,
                legacyDownNanos / 1_000_000.0 / repetitions, refDownNanos / 1_000_000.0 / repetitions);
    }

    private static AbstractTensor legacyMlp(NativeSimdTensorOperations ops, AbstractTensor input,
            AbstractTensor gateWeight, AbstractTensor upWeight, AbstractTensor downWeight, int embedding, int hidden) {
        AbstractTensor gate = new FloatBufferTensor(1, hidden);
        AbstractTensor up = new FloatBufferTensor(1, hidden);
        AbstractTensor output = new FloatBufferTensor(1, embedding);
        try {
            ops.dotProductBatchChunk(new AbstractTensor[]{gate, up}, input,
                    new AbstractTensor[]{gateWeight, upWeight}, 0, embedding, 0, hidden);
            AbstractTensor hiddenInput = ops.activationMultiplyQuantize(gate, up, ActivationFunction.Type.SILU,
                    DType.I8, 0, hidden);
            try {
                ops.dotProductChunk(output, hiddenInput, downWeight, 0, hidden, 0, embedding);
            } finally {
                hiddenInput.close();
            }
            return output;
        } finally {
            gate.close();
            up.close();
        }
    }

    private static FloatBufferTensor tensor(int rows, int columns, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, columns);
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                tensor.set(((row * 13 + column * 7 + seed) % 17 - 8) / 8.0f, row, column);
            }
        }
        return tensor;
    }
}
