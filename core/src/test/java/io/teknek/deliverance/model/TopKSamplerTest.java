package io.teknek.deliverance.model;

import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TopKSamplerTest {

    @Test
    public void topKGreaterThanOneIsTokenCount() {
        assertEquals(64, DeliveranceSampler.topKCandidateCount(64.0f, 262_144));
    }

    @Test
    public void fractionalTopKPreservesLegacyCutoffBehavior() {
        assertEquals(90, DeliveranceSampler.topKCandidateCount(0.10f, 100));
    }

    @Test
    public void combinedTopKAndTopPSamplesFromFilteredCandidates() {
        FloatBufferTensor logits = new FloatBufferTensor(1, 5);
        logits.set(5.0f, 0, 0);
        logits.set(4.0f, 0, 1);
        logits.set(3.0f, 0, 2);
        logits.set(2.0f, 0, 3);
        logits.set(1.0f, 0, 4);

        int picked = DeliveranceSampler.sampleTopKTopP(logits, 1.0f, Optional.of(2.0f), Optional.of(0.95f),
                Optional.empty(), 0.99f);

        assertEquals(1, picked);
    }

    @Test
    public void uniformTopPSamplesUniformlyFromNucleus() {
        FloatBufferTensor logits = new FloatBufferTensor(1, 5);
        logits.set(5.0f, 0, 0);
        logits.set(4.0f, 0, 1);
        logits.set(3.0f, 0, 2);
        logits.set(2.0f, 0, 3);
        logits.set(1.0f, 0, 4);

        int weighted = DeliveranceSampler.sampleTopKTopP(logits, 1.0f, Optional.of(2.0f), Optional.of(0.95f),
                Optional.empty(), 0.60f);
        int uniform = DeliveranceSampler.sampleTopKTopP(logits, 1.0f, Optional.of(2.0f), Optional.empty(),
                Optional.of(0.95f), 0.60f);

        assertEquals(0, weighted);
        assertEquals(1, uniform);
    }

    @Test
    public void uniformTopPRejectsInvalidValues() {
        FloatBufferTensor logits = new FloatBufferTensor(1, 2);
        logits.set(2.0f, 0, 0);
        logits.set(1.0f, 0, 1);

        assertThrows(IllegalArgumentException.class, () -> DeliveranceSampler.sampleTopKTopP(logits, 1.0f,
                Optional.empty(), Optional.empty(), Optional.of(0.0f), 0.50f));
        assertThrows(IllegalArgumentException.class, () -> DeliveranceSampler.sampleTopKTopP(logits, 1.0f,
                Optional.empty(), Optional.empty(), Optional.of(1.01f), 0.50f));
        assertThrows(IllegalArgumentException.class, () -> DeliveranceSampler.sampleTopKTopP(logits, 1.0f,
                Optional.empty(), Optional.of(0.95f), Optional.of(0.95f), 0.50f));
    }
}
