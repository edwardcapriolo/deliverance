package io.teknek.deliverance.tensorlib;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorTestSupport;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
