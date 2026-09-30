package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterMaxTest {
    @Test
    void panamaWritesMaximumToScalarTensor() {
        Lighter lighter = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.PANAMA, new PanamaOps()));
        try (TensorRef input = lighter.allocate(DType.F32, TensorShape.of(2, 65));
             TensorRef output = lighter.allocate(DType.F32, TensorShape.of(1, 1))) {
            for (int column = 0; column < 65; column++) {
                input.set(column == 47 ? 99.0f : -column, 1, column);
            }
            lighter.max(new Max(input).into(output).row(1).offsetAndLength(3, 50));
            assertEquals(99.0f, output.get(0, 0));
        }
    }
}
