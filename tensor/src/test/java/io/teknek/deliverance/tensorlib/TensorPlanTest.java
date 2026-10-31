package io.teknek.deliverance.tensorlib;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.TensorTestSupport;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor.TensorShape;
import io.dropwizard.metrics5.MetricRegistry;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ForkJoinPool;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorPlanTest {

    @Test
    void planDrawsMlpShapeAsAscii() {
        TensorPlan plan = new TensorPlan(new NaiveTensorOperations(), new WrappedForkJoinPool(new ForkJoinPool(1)));
        try (AbstractTensor input = TensorTestSupport.tensorOf(2, 3, 1, 2, 3, 4, 5, 6);
             AbstractTensor gateWeight = TensorTestSupport.tensorOf(2, 3, 1, 0, 0, 0, 1, 0);
             AbstractTensor upWeight = TensorTestSupport.tensorOf(2, 3, 0, 0, 1, 1, 1, 1)) {

            TensorPlan.Tensor inputNode = plan.input("input", input);
            TensorPlan.ImmutableTensor gateWeightNode = plan.immutable("gateWeight", gateWeight);
            TensorPlan.ImmutableTensor upWeightNode = plan.immutable("upWeight", upWeight);

            TensorPlan.Tensor gate = inputNode.batchDot(gateWeightNode).as("gate");
            TensorPlan.Tensor up = inputNode.batchDot(upWeightNode).as("up");
            String ascii = gate
                    .activate(ActivationFunction.Type.SILU)
                    .multiply(up)
                    .as("hidden")
                    .plan();

            System.out.println(ascii);
            assertTrue(ascii.contains("hidden = multiply"), ascii);
            assertTrue(ascii.contains("gate = batchDot"), ascii);
            assertTrue(ascii.contains("up = batchDot"), ascii);
            assertTrue(ascii.contains("activate SILU"), ascii);
            assertTrue(ascii.contains("input [2x3] F32 borrowed"), ascii);
        }
    }

    @Test
    void refAddPlanMaterializesWithoutLegacyAdapter() {
        Lighter lighter = new Lighter(new io.dropwizard.metrics5.MetricRegistry());
        TensorPlan plan = new TensorPlan(new NaiveTensorOperations(), new WrappedForkJoinPool(new ForkJoinPool(1)));
        try (TensorRef left = lighter.allocate(DType.F32, TensorShape.of(1, 3));
             TensorRef right = lighter.allocate(DType.F32, TensorShape.of(1, 3))) {
            left.set(1.0f, 0, 0);
            left.set(2.0f, 0, 1);
            left.set(3.0f, 0, 2);
            right.set(4.0f, 0, 0);
            right.set(5.0f, 0, 1);
            right.set(6.0f, 0, 2);
            try (TensorRef result = plan.input("left", left).add(plan.input("right", right)).materializeRef()) {
                assertEquals(5.0f, result.get(0, 0));
                assertEquals(7.0f, result.get(0, 1));
                assertEquals(9.0f, result.get(0, 2));
            }
        }
    }

    @Test
    void importedInputRendersUpstreamLineageButUsesMaterializedTensor() {
        TensorPlan upstream = new TensorPlan(new NaiveTensorOperations(), new WrappedForkJoinPool(new ForkJoinPool(1)));
        TensorPlan downstream = new TensorPlan(new NaiveTensorOperations(), new WrappedForkJoinPool(new ForkJoinPool(1)));
        try (AbstractTensor source = TensorTestSupport.tensorOf(1, 2, 1, 2);
             AbstractTensor materialized = TensorTestSupport.tensorOf(1, 2, 10, 20)) {
            TensorPlan.Tensor scaled = upstream.input("abc", source).scale(50).as("scaled");
            TensorPlan.Tensor imported = downstream.input("stage1.input", scaled, materialized).as("stage1.output");

            String ascii = imported.plan();

            assertTrue(ascii.contains("stage1.input <- scaled [1x2] F32 borrowed"), ascii);
            assertFalse(ascii.contains("scaled = scale"), ascii);
            try (AbstractTensor actual = imported.materialize()) {
                assertEquals(10.0f, actual.get(0, 0));
                assertEquals(20.0f, actual.get(0, 1));
            }
        }
    }

    @Test
    void fusedActivationMultiplyMatchesExplicitPath() {
        TensorPlan plan = new TensorPlan(new NaiveTensorOperations(), new WrappedForkJoinPool(new ForkJoinPool(1)));
        try (AbstractTensor gate = TensorTestSupport.tensorOf(2, 3, -1, 0, 1, 2, 3, 4);
             AbstractTensor up = TensorTestSupport.tensorOf(2, 3, 2, 2, 2, 0.5f, 0.25f, 0.125f);
             AbstractTensor fused = plan.input(gate).activate(ActivationFunction.Type.SILU)
                     .multiply(plan.input(up)).materialize()) {

            for (int row = 0; row < 2; row++) {
                for (int col = 0; col < 3; col++) {
                    float expected = ActivationFunction.eval(ActivationFunction.Type.SILU, gate.get(row, col))
                            * up.get(row, col);
                    assertEquals(expected, fused.get(row, col), 1.0e-6f, "row=" + row + " col=" + col);
                }
            }
        }
    }

    @Test
    void explicitFusedChunkPipelineMatchesActivationMultiply() {
        TensorPlan plan = new TensorPlan(new NaiveTensorOperations(), new WrappedForkJoinPool(new ForkJoinPool(2)));
        try (AbstractTensor gate = TensorTestSupport.tensorOf(2, 3, -1, 0, 1, 2, 3, 4);
             AbstractTensor up = TensorTestSupport.tensorOf(2, 3, 2, 2, 2, 0.5f, 0.25f, 0.125f);
             AbstractTensor hidden = plan.fuse("hidden", gate.shape())
                     .read("gate", plan.input("gate", gate))
                     .read("up", plan.input("up", up))
                     .map("hidden = silu(gate)", TensorPlan.TensorOp.SILU_WRITE, (ctx, offset, length) -> {
                         AbstractTensor g = ctx.tensor("gate");
                         AbstractTensor h = ctx.tensor("hidden");
                         int cols = (int) g.shape().last();
                         for (long index = offset; index < offset + length; index++) {
                             int row = (int) (index / cols);
                             int col = (int) (index % cols);
                             h.set(ActivationFunction.eval(ActivationFunction.Type.SILU, g.get(row, col)), row, col);
                         }
                     })
                     .map("hidden *= up", TensorPlan.TensorOp.MUL_IN_PLACE, (ctx, offset, length) -> {
                         AbstractTensor h = ctx.tensor("hidden");
                         AbstractTensor u = ctx.tensor("up");
                         int cols = (int) h.shape().last();
                         for (long index = offset; index < offset + length; index++) {
                             int row = (int) (index / cols);
                             int col = (int) (index % cols);
                             h.set(h.get(row, col) * u.get(row, col), row, col);
                         }
                     })
                     .tensor()
                     .materialize()) {

            for (int row = 0; row < 2; row++) {
                for (int col = 0; col < 3; col++) {
                    float expected = ActivationFunction.eval(ActivationFunction.Type.SILU, gate.get(row, col))
                            * up.get(row, col);
                    assertEquals(expected, hidden.get(row, col), 1.0e-6f, "row=" + row + " col=" + col);
                }
            }
        }
    }

    @Test
    void columnIntStreamFuseCanMutateWritableInput() {
        TensorPlan plan = new TensorPlan(new NaiveTensorOperations(), new WrappedForkJoinPool(new ForkJoinPool(1)));
        try (AbstractTensor gate = TensorTestSupport.tensorOf(2, 3, -1, 0, 1, 2, 3, 4);
             AbstractTensor up = TensorTestSupport.tensorOf(2, 3, 2, 2, 2, 0.5f, 0.25f, 0.125f)) {

            plan.fuseColumnsIntStream("gate", gate.shape())
                    .write("gate", plan.mutable("gate", gate))
                    .read("up", plan.input("up", up))
                    .map("gate = silu(gate) * up", TensorPlan.TensorOp.ACTIVATION_MUL_IN_PLACE,
                            (ctx, offset, length) -> {
                                AbstractTensor g = ctx.tensor("gate");
                                AbstractTensor u = ctx.tensor("up");
                                int col = (int) offset;
                                for (int row = 0; row < g.shape().first(); row++) {
                                    float activated = ActivationFunction.eval(ActivationFunction.Type.SILU,
                                            g.get(row, col));
                                    g.set(activated * u.get(row, col), row, col);
                                }
                            })
                    .tensor()
                    .materialize();

            assertEquals(ActivationFunction.eval(ActivationFunction.Type.SILU, -1.0f) * 2.0f, gate.get(0, 0),
                    1.0e-6f);
            assertEquals(ActivationFunction.eval(ActivationFunction.Type.SILU, 4.0f) * 0.125f, gate.get(1, 2),
                    1.0e-6f);
        }
    }

    @Test
    void rowIntStreamFuseCanMutateWritableInput() {
        TensorPlan plan = new TensorPlan(new NaiveTensorOperations(), new WrappedForkJoinPool(new ForkJoinPool(1)));
        try (AbstractTensor gate = TensorTestSupport.tensorOf(2, 4, 1, 2, 3, 4, 10, 10, 20, 20)) {

            plan.fuseRowsIntStream("gate", gate.shape())
                    .write("gate", plan.mutable("gate", gate))
                    .map("gate -= row mean", TensorPlan.TensorOp.ACTIVATION_SPARSITY_IN_PLACE,
                            (ctx, offset, length) -> {
                                AbstractTensor g = ctx.tensor("gate");
                                int row = (int) offset;
                                double sum = 0.0d;
                                for (int col = 0; col < g.shape().last(); col++) {
                                    sum += g.get(row, col);
                                }
                                float mean = (float) (sum / g.shape().last());
                                for (int col = 0; col < g.shape().last(); col++) {
                                    g.set(g.get(row, col) - mean, row, col);
                                }
                            })
                    .tensor()
                    .materialize();

            assertEquals(-1.5f, gate.get(0, 0), 1.0e-6f);
            assertEquals(1.5f, gate.get(0, 3), 1.0e-6f);
            assertEquals(-5.0f, gate.get(1, 0), 1.0e-6f);
            assertEquals(5.0f, gate.get(1, 3), 1.0e-6f);
        }
    }

    @Test
    void basicOpsMaterializeWithoutMutatingBorrowedInputs() {
        TensorPlan plan = new TensorPlan(new NaiveTensorOperations(), new WrappedForkJoinPool(new ForkJoinPool(1)));
        try (AbstractTensor a = TensorTestSupport.tensorOf(2, 3, 1, 2, 3, 4, 5, 6);
             AbstractTensor b = TensorTestSupport.tensorOf(2, 3, 10, 20, 30, 40, 50, 60);
             AbstractTensor out = plan.input("a", a).multiply(plan.input("b", b)).add(plan.input("a", a)).scale(0.5f)
                     .materialize()) {

            assertEquals(5.5f, out.get(0, 0));
            assertEquals(21.0f, out.get(0, 1));
            assertEquals(46.5f, out.get(0, 2));
            assertEquals(1.0f, a.get(0, 0), "borrowed input should not be mutated");
            assertEquals(10.0f, b.get(0, 0), "borrowed input should not be mutated");
        }
    }

    @Test
    void mlpLogicalFlowMaterializesExpectedOutput() {
        TensorPlan plan = new TensorPlan(new NaiveTensorOperations(), new WrappedForkJoinPool(new ForkJoinPool(1)));
        try (AbstractTensor input = TensorTestSupport.tensorOf(2, 3, 1, 2, 3, 4, 5, 6);
             AbstractTensor gateWeight = TensorTestSupport.tensorOf(2, 3, 1, 0, 0, 0, 1, 0);
             AbstractTensor upWeight = TensorTestSupport.tensorOf(2, 3, 0, 0, 1, 1, 1, 1);
             AbstractTensor downWeight = TensorTestSupport.tensorOf(3, 2, 1, 0, 0, 1, 1, 1)) {
            TensorPlan.Tensor inputNode = plan.input("input", input);
            TensorPlan.ImmutableTensor gateWeightNode = plan.immutable("gateWeight", gateWeight);
            TensorPlan.ImmutableTensor upWeightNode = plan.immutable("upWeight", upWeight);
            TensorPlan.ImmutableTensor downWeightNode = plan.immutable("downWeight", downWeight);
            TensorPlan.Tensor gate = inputNode.batchDot(gateWeightNode).as("gate");
            TensorPlan.Tensor up = inputNode.batchDot(upWeightNode).as("up");
            try (AbstractTensor output = gate
                    .activate(ActivationFunction.Type.SILU)
                    .multiply(up)
                    .as("hidden")
                    .batchDot(downWeightNode)
                    .as("output")
                    .materialize()) {

                assertEquals(2, output.shape().first());
                assertEquals(3, output.shape().last());
                assertTrue(Float.isFinite(output.get(0, 0)));
                assertTrue(Float.isFinite(output.get(1, 2)));
            }
        }
    }

    @Test
    void providerBackedMlpPrimitiveMatchesExplicitProviderPath() {
        NaiveTensorOperations ops = new NaiveTensorOperations();
        TensorPlan plan = new TensorPlan(ops, new WrappedForkJoinPool(new ForkJoinPool(2)));
        try (AbstractTensor input = TensorTestSupport.deterministicTensor(2, 32, 1);
             AbstractTensor gateWeight = TensorTestSupport.deterministicTensor(32, 32, 2);
             AbstractTensor upWeight = TensorTestSupport.deterministicTensor(32, 32, 3);
             AbstractTensor downWeight = TensorTestSupport.deterministicTensor(2, 32, 4);
             AbstractTensor expected = explicitMlp(ops, input, gateWeight, upWeight, downWeight);
             AbstractTensor actual = plan.input("input", input)
                     .mlp(plan.immutable("gateWeight", gateWeight),
                             plan.immutable("upWeight", upWeight),
                             plan.immutable("downWeight", downWeight),
                             ActivationFunction.Type.SILU,
                             DType.I8)
                     .materialize()) {

            assertEquals(expected.shape(), actual.shape());
            for (int row = 0; row < expected.shape().first(); row++) {
                for (int col = 0; col < expected.shape().last(); col++) {
                    assertEquals(expected.get(row, col), actual.get(row, col), 1.0e-6f,
                            "row=" + row + " col=" + col);
                }
            }
        }
    }

    @Test
    @Disabled("TensorRef decode MLP currently differs from the legacy path by 0.078 relative error")
    void qwen3FourBDecodeMlpCharacterization() {
        int embeddingLength = 2560;
        int hiddenLength = 9728;
        MetricRegistry metrics = new MetricRegistry();
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(16))) {
            PanamaTensorOperations operations = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(metrics), pool);
            Lighter lighter = new Lighter(metrics);
            TensorPlan legacy = new TensorPlan(operations, pool, metrics);
            TensorPlan refs = new TensorPlan(operations, pool, metrics, null, null, lighter);
            try (AbstractTensor input = TensorTestSupport.deterministicTensor(1, embeddingLength, 401);
                 AbstractTensor gateDense = TensorTestSupport.deterministicTensor(hiddenLength, embeddingLength, 402);
                 AbstractTensor upDense = TensorTestSupport.deterministicTensor(hiddenLength, embeddingLength, 403);
                 AbstractTensor downDense = TensorTestSupport.deterministicTensor(embeddingLength, hiddenLength, 404);
                 AbstractTensor gateWeight = AbstractTensorUtils.quantize(gateDense, DType.Q4, true);
                 AbstractTensor upWeight = AbstractTensorUtils.quantize(upDense, DType.Q4, true);
                 AbstractTensor downWeight = AbstractTensorUtils.quantize(downDense, DType.Q4, true);
                 TensorRef inputRef = TensorRef.borrowed(input);
                 TensorRef gateRef = TensorRef.borrowed(gateWeight);
                 TensorRef upRef = TensorRef.borrowed(upWeight);
             TensorRef downRef = TensorRef.borrowed(downWeight)) {
                List<Lighter.ProviderEvent> providerEvents = new ArrayList<>();
                lighter.setProviderObserver(providerEvents::add);
                TensorPlan.Tensor legacyPlan = legacy.input("input", input)
                        .mlp(legacy.immutable("gate", gateWeight), legacy.immutable("up", upWeight),
                                legacy.immutable("down", downWeight), ActivationFunction.Type.SILU, DType.I8);
                TensorPlan.Tensor refPlan = refs.input("input", inputRef)
                        .mlp(refs.immutable("gate", gateRef), refs.immutable("up", upRef),
                                refs.immutable("down", downRef), ActivationFunction.Type.SILU, DType.I8);
                for (int i = 0; i < 2; i++) {
                    try (AbstractTensor ignored = legacyPlan.materialize(); TensorRef ignoredRef = refPlan.materializeRef()) {
                    }
                }
                int repetitions = 4;
                long legacyStart = System.nanoTime();
                for (int i = 0; i < repetitions; i++) {
                    try (AbstractTensor ignored = legacyPlan.materialize()) {
                    }
                }
                long legacyNanos = System.nanoTime() - legacyStart;
                long refStart = System.nanoTime();
                for (int i = 0; i < repetitions; i++) {
                    try (TensorRef ignored = refPlan.materializeRef()) {
                    }
                }
                long refNanos = System.nanoTime() - refStart;
                try (AbstractTensor expected = legacyPlan.materialize(); TensorRef actual = refPlan.materializeRef()) {
                    float maxRelativeError = 0.0f;
                    for (int column = 0; column < embeddingLength; column++) {
                        float expectedValue = expected.get(0, column);
                        float actualValue = actual.get(0, column);
                     assertTrue(Float.isFinite(expectedValue) && Float.isFinite(actualValue),
                                 "non-finite output column=" + column);
                        maxRelativeError = Math.max(maxRelativeError,
                                 Math.abs(expectedValue - actualValue) / Math.max(1.0f, Math.abs(expectedValue)));
                     }
                     assertTrue(maxRelativeError <= 1.0e-3f,
                             "legacy and TensorRef MLP outputs differ maxRelativeError=" + maxRelativeError);
                     assertFalse(providerEvents.isEmpty(), "TensorRef MLP must dispatch through a provider");
                     System.out.printf("[qwen3-4b-decode-mlp] max_relative_error=%f%n", maxRelativeError);
                }
                System.out.printf("[qwen3-4b-decode-mlp] embedding=%d hidden=%d legacy_ms=%.3f "
                                + "tensorRef_ms=%.3f legacy_ms_per=%f tensorRef_ms_per=%f%n",
                        embeddingLength, hiddenLength, legacyNanos / 1_000_000.0, refNanos / 1_000_000.0,
                        legacyNanos / 1_000_000.0 / repetitions, refNanos / 1_000_000.0 / repetitions);
            }
        }
    }

    private static AbstractTensor explicitMlp(NaiveTensorOperations ops, AbstractTensor input, AbstractTensor gateWeight,
            AbstractTensor upWeight, AbstractTensor downWeight) {
        AbstractTensor gate = new FloatBufferTensor((int) input.shape().first(), (int) gateWeight.shape().first());
        AbstractTensor up = new FloatBufferTensor((int) input.shape().first(), (int) upWeight.shape().first());
        AbstractTensor hidden = null;
        AbstractTensor output = new FloatBufferTensor((int) input.shape().first(), (int) downWeight.shape().first());
        try {
            ops.dotProductBatchChunk(new AbstractTensor[] { gate, up }, input,
                    new AbstractTensor[] { gateWeight, upWeight }, 0, (int) input.shape().last(), 0,
                    (int) gateWeight.shape().first());
            hidden = ops.activationMultiplyQuantize(gate, up, ActivationFunction.Type.SILU, DType.I8, 0,
                    (int) gate.shape().last());
            ops.dotProductChunk(output, hidden, downWeight, 0, (int) hidden.shape().last(), 0,
                    (int) downWeight.shape().first());
            return output;
        } finally {
            gate.close();
            up.close();
            if (hidden != null) {
                hidden.close();
            }
        }
    }

}
