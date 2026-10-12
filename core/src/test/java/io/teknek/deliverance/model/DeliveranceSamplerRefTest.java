package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.GeneratorParameters;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DeliveranceSamplerRefTest {
    @Test
    void defaultSamplingUsesLegacyTopP() {
        GeneratorParameters parameters = new GeneratorParameters().withTemperature(1.0f);
        if (parameters.xtcThreshold.isEmpty() && parameters.topK.isEmpty()
                && parameters.topP.isEmpty() && parameters.uniformTopP.isEmpty()) {
            parameters.topP = java.util.Optional.of(0.10f);
        }
        DeliveranceSamplerRef.validateSamplingParameters(parameters, 1.0f);
        assertEquals(0.10f, parameters.topP.orElseThrow());
    }

    @Test
    void invalidSamplingCombinationsMatchLegacyValidation() {
        assertThrows(IllegalArgumentException.class, () ->
                DeliveranceSamplerRef.validateSamplingParameters(
                        new GeneratorParameters().withTemperature(1.0f).withTopP(0.0f), 1.0f));
        assertThrows(IllegalArgumentException.class, () ->
                DeliveranceSamplerRef.validateSamplingParameters(
                        new GeneratorParameters().withTemperature(1.0f).withUniformTopP(0.5f), 1.0f));
        assertThrows(IllegalArgumentException.class, () ->
                DeliveranceSamplerRef.validateSamplingParameters(
                        new GeneratorParameters().withTemperature(1.0f).withTopK(2).withXtcThreshold(0.1f), 1.0f));
        assertThrows(IllegalArgumentException.class, () ->
                DeliveranceSamplerRef.validateSamplingParameters(
                        new GeneratorParameters().withTemperature(0.0f).withUniformTopP(0.5f), 0.0f));
    }
}
