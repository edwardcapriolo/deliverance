package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterArgMaxTest {
    @Test
    void panamaMatchesNaiveAndChoosesLowestTieIndex() {
        Lighter expected = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.NAIVE, new NaiveOps()));
        Lighter actual = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.PANAMA, new PanamaOps()));
        try (TensorRef expectedInput = expected.allocate(DType.F32, TensorShape.of(1, 65));
             TensorRef actualInput = actual.allocate(DType.F32, TensorShape.of(1, 65));
             TensorRef expectedOutput = expected.allocate(DType.F32, TensorShape.of(1, 2));
             TensorRef actualOutput = actual.allocate(DType.F32, TensorShape.of(1, 2))) {
            for (int column = 0; column < 65; column++) {
                float value = column == 17 || column == 49 ? 12.5f : -1.0f;
                expectedInput.set(value, 0, column);
                actualInput.set(value, 0, column);
            }
            expected.argMax(new ArgMax(expectedInput).into(expectedOutput).offsetAndLength(3, 50));
            actual.argMax(new ArgMax(actualInput).into(actualOutput).offsetAndLength(3, 50));
            assertEquals(expectedOutput.get(0, 0), actualOutput.get(0, 0));
            assertEquals(expectedOutput.get(0, 1), actualOutput.get(0, 1));
            assertEquals(17.0f, actualOutput.get(0, 0));
            assertEquals(12.5f, actualOutput.get(0, 1));
        }
    }
}
