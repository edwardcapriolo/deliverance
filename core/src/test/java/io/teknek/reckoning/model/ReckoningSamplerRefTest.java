package io.teknek.reckoning.model;

import io.teknek.reckoning.generator.GeneratorParameters;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ReckoningSamplerRefTest {
    @Test
    void defaultSamplingUsesLegacyTopP() {
        GeneratorParameters parameters = new GeneratorParameters().withTemperature(1.0f);
        if (parameters.xtcThreshold.isEmpty() && parameters.topK.isEmpty()
                && parameters.topP.isEmpty() && parameters.uniformTopP.isEmpty()) {
            parameters.topP = java.util.Optional.of(0.10f);
        }
        ReckoningSamplerRef.validateSamplingParameters(parameters, 1.0f);
        assertEquals(0.10f, parameters.topP.orElseThrow());
    }

    @Test
    void invalidSamplingCombinationsMatchLegacyValidation() {
        assertThrows(IllegalArgumentException.class, () ->
                ReckoningSamplerRef.validateSamplingParameters(
                        new GeneratorParameters().withTemperature(1.0f).withTopP(0.0f), 1.0f));
        assertThrows(IllegalArgumentException.class, () ->
                ReckoningSamplerRef.validateSamplingParameters(
                        new GeneratorParameters().withTemperature(1.0f).withUniformTopP(0.5f), 1.0f));
        assertThrows(IllegalArgumentException.class, () ->
                ReckoningSamplerRef.validateSamplingParameters(
                        new GeneratorParameters().withTemperature(1.0f).withTopK(2).withXtcThreshold(0.1f), 1.0f));
        assertThrows(IllegalArgumentException.class, () ->
                ReckoningSamplerRef.validateSamplingParameters(
                        new GeneratorParameters().withTemperature(0.0f).withUniformTopP(0.5f), 0.0f));
    }

    @Test
    void topLogProbsContainNormalizedLogProbabilities() {
        try (TensorRef logits = new Lighter().allocate(DType.F32, TensorShape.of(1, 4))) {
            logits.set(1.0f, 0, 0);
            logits.set(4.0f, 0, 1);
            logits.set(2.0f, 0, 2);
            logits.set(-1.0f, 0, 3);

            var top = ReckoningSamplerRef.topLogProbs(logits, 2, index -> "token-" + index);
            var entries = top.stream().sorted(java.util.Comparator.comparingDouble((IndexValueToken value) -> value.value)
                    .reversed()).toList();
            double normalizer = Math.log(Math.exp(1.0) + Math.exp(4.0) + Math.exp(2.0) + Math.exp(-1.0));

            assertEquals(2, entries.size());
            assertEquals(1, entries.get(0).index);
            assertEquals(4.0f - normalizer, entries.get(0).logProb, 1.0e-6f);
            assertEquals(2, entries.get(1).index);
            assertEquals(2.0f - normalizer, entries.get(1).logProb, 1.0e-6f);
            assertTrue(entries.stream().allMatch(value -> value.getProbability() > 0.0f));
        }
    }
}
