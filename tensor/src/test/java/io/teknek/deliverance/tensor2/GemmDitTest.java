package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class GemmDitTest {
    @Test
    void alignedQ4WorkOffersChunkCandidates() {
        List<String> candidates = GemmDit.candidateNames(DType.I8, DType.Q4, 0, 32, 64);

        assertTrue(candidates.contains("CHUNK/NONE"));
        assertTrue(candidates.contains("CHUNK/PROVIDER"));
    }

    @Test
    void unalignedQ4WorkOffersDedicatedChunkCandidates() {
        List<String> candidates = GemmDit.candidateNames(DType.I8, DType.Q4, 1, 32, 64);

        assertTrue(candidates.stream().anyMatch(name -> name.startsWith("CHUNK/")));
        assertTrue(candidates.contains("ROW/NONE"));
        assertTrue(candidates.contains("ROW/PROVIDER"));
    }
}
