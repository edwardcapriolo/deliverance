package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.tensorlib.TensorPlan;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorPlanLighterTest {

    @Test
    void tensorPlanChunksMultiplyAccumulateThroughLighter() throws Exception {
        Lighter expected = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.NAIVE, new NaiveOps()));
        Lighter actual = new Lighter(new MetricRegistry(), Map.of(
                TensorProviderKind.PANAMA, new PanamaOps(),
                TensorProviderKind.NAIVE, new NaiveOps()
        ));
        TensorRef expectedA = expected.allocate(DType.F32, TensorShape.of(2, 16));
        TensorRef expectedB = expected.allocate(DType.F32, TensorShape.of(2, 16));
        TensorRef actualA = actual.allocate(DType.F32, TensorShape.of(2, 16));
        TensorRef actualB = actual.allocate(DType.F32, TensorShape.of(2, 16));
        fill(expectedA);
        fill(expectedB);
        fill(actualA);
        fill(actualB);

        expected.multiplyAccumulate(new MultiplyAccumulate(expectedB).into(expectedA).offsetAndLength(0, 16));
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(2))) {
            TensorPlan plan = new TensorPlan(new NaiveTensorOperations(), pool);
            plan.chunked("tensor2.multiply_accumulate", 0, 16)
                    .splitCount(4)
                    .run((offset, length) ->
                            actual.multiplyAccumulate(new MultiplyAccumulate(actualB).into(actualA)
                                    .offsetAndLength((int) offset, (int) length)));
        }

        for (int row = 0; row < 2; row++) {
            for (int column = 0; column < 16; column++) {
                assertEquals(expectedA.underlying().get(row, column), actualA.underlying().get(row, column), 0.0f,
                        "row=" + row + " column=" + column);
            }
        }
    }

    private static void fill(TensorRef tensor) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.underlying().set(row * 17.0f + column + 1.0f, row, column);
            }
        }
    }
}
