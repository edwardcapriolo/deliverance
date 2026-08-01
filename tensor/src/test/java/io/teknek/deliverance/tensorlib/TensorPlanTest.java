package io.teknek.deliverance.tensorlib;

import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorTestSupport;
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
}
