package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorProbability;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorProbabilityTensorRefTest {
    @Test
    void tensorRefEntropyMatchesScalarReference() {
        Lighter lighter = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.PANAMA, new PanamaOps()));
        try (TensorRef logits = lighter.allocate(DType.F32, TensorShape.of(2, 3, 17));
             TensorRef entropy = lighter.allocate(DType.F32, TensorShape.of(2, 3))) {
            for (int batch = 0; batch < 2; batch++) {
                for (int position = 0; position < 3; position++) {
                    for (int token = 0; token < 17; token++) {
                        logits.set(((batch * 23 + position * 11 + token * 7) % 31 - 15) / 5.0f,
                                batch, position, token);
                    }
                }
            }
            TensorProbability.entropy(entropy, logits, lighter);
            for (int batch = 0; batch < 2; batch++) {
                for (int position = 0; position < 3; position++) {
                    assertEquals(scalarEntropy(logits, batch, position), entropy.get(batch, position), 1.0e-5f,
                            "batch=" + batch + " position=" + position);
                }
            }
        }
    }

    private static double scalarEntropy(TensorRef logits, int batch, int position) {
        float maximum = Float.NEGATIVE_INFINITY;
        for (int token = 0; token < logits.shape().dim(2); token++) {
            maximum = Math.max(maximum, logits.get(batch, position, token));
        }
        double sumExp = 0.0;
        double weighted = 0.0;
        for (int token = 0; token < logits.shape().dim(2); token++) {
            double shifted = logits.get(batch, position, token) - maximum;
            double value = Math.exp(shifted);
            sumExp += value;
            weighted += value * shifted;
        }
        return Math.log(sumExp) - weighted / sumExp;
    }
}
