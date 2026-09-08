package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.tensorlib.TensorPlan;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorPlanLighterCharacterizationTest {

    @Test
    void characterizeChunkedMultiplyAccumulate() {
        int rows = 16;
        int columns = 4096;
        Lighter oneShot = new Lighter(new MetricRegistry(), Map.of(
                TensorProviderKind.PANAMA, new PanamaOps(),
                TensorProviderKind.NAIVE, new NaiveOps()
        ));
        Lighter chunked = new Lighter(new MetricRegistry(), Map.of(
                TensorProviderKind.PANAMA, new PanamaOps(),
                TensorProviderKind.NAIVE, new NaiveOps()
        ));
        TensorRef oneShotA = oneShot.allocate(DType.F32, TensorShape.of(rows, columns));
        TensorRef oneShotB = oneShot.allocate(DType.F32, TensorShape.of(rows, columns));
        TensorRef chunkedA = chunked.allocate(DType.F32, TensorShape.of(rows, columns));
        TensorRef chunkedB = chunked.allocate(DType.F32, TensorShape.of(rows, columns));
        fill(oneShotA, oneShotB, chunkedA, chunkedB);

        long oneShotStart = System.nanoTime();
        oneShot.multiplyAccumulate(new MultiplyAccumulate(oneShotB).into(oneShotA).offsetAndLength(0, columns));
        long oneShotNanos = System.nanoTime() - oneShotStart;

        long chunkedNanos;
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(8))) {
            TensorPlan plan = new TensorPlan(new NaiveTensorOperations(), pool);
            long chunkedStart = System.nanoTime();
            plan.chunked("tensor2.multiply_accumulate", 0, columns)
                    .run((offset, length) -> chunked.multiplyAccumulate(new MultiplyAccumulate(chunkedB).into(chunkedA)
                            .offsetAndLength((int) offset, (int) length)));
            chunkedNanos = System.nanoTime() - chunkedStart;
        }

        assertClose(oneShotA, chunkedA, rows, columns);
        System.out.printf("tensor2.multiply_accumulate rows=%d columns=%d one_shot_ms=%.3f chunked_ms=%.3f%n",
                rows, columns, oneShotNanos / 1_000_000.0, chunkedNanos / 1_000_000.0);
    }

    private static void fill(TensorRef oneShotA, TensorRef oneShotB, TensorRef chunkedA, TensorRef chunkedB) {
        Random random = new Random(0x5eed);
        for (int row = 0; row < oneShotA.shape().first(); row++) {
            for (int column = 0; column < oneShotA.shape().last(); column++) {
                float a = random.nextFloat() - 0.5f;
                float b = random.nextFloat() - 0.5f;
                oneShotA.underlying().set(a, row, column);
                chunkedA.underlying().set(a, row, column);
                oneShotB.underlying().set(b, row, column);
                chunkedB.underlying().set(b, row, column);
            }
        }
    }

    private static void assertClose(TensorRef expected, TensorRef actual, int rows, int columns) {
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                assertEquals(expected.underlying().get(row, column), actual.underlying().get(row, column), 0.0f,
                        "row=" + row + " column=" + column);
            }
        }
    }
}
