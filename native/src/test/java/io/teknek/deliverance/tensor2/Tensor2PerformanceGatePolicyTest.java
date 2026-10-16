package io.teknek.deliverance.tensor2;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Tensor2PerformanceGatePolicyTest {
    @Test
    void valueWithinTenPercentOfHistoricalMaximumPassesWithoutRetries() {
        Tensor2PerformanceGatePolicy.Decision decision = evaluate(1_000, 1_090, List.of());

        assertTrue(decision.passed());
        assertTrue(decision.suspects().isEmpty());
    }

    @Test
    void firstStageSuspectRecoversOnFirstRetry() {
        Tensor2PerformanceGatePolicy.Decision decision = evaluate(1_000, 1_500, List.of(950L));

        assertTrue(decision.passed());
        assertEquals(1, decision.suspects().size());
    }

    @ParameterizedTest(name = "retry {0} is recovered")
    @MethodSource("fasterOrEqualRetries")
    void retryAtOrBelowHistoricalMaximumRecovers(long retryValue) {
        Tensor2PerformanceGatePolicy.Decision decision = evaluate(1_000, 1_500, List.of(retryValue));

        assertTrue(decision.passed());
    }

    @Test
    void persistentSlowdownFailsAfterTenRetries() {
        Tensor2PerformanceGatePolicy.Decision decision = evaluate(1_000, 1_500,
                List.of(1_101L, 1_102L, 1_103L, 1_104L, 1_105L, 1_106L, 1_107L, 1_108L, 1_109L, 1_110L,
                        1L));

        assertFalse(decision.passed());
        assertEquals(1, decision.failures().size());
        assertEquals(10, decision.failures().get(0).retryValues().size());
    }

    static Stream<Arguments> fasterOrEqualRetries() {
        return Stream.of(Arguments.of(999L), Arguments.of(1_000L));
    }

    private static Tensor2PerformanceGatePolicy.Decision evaluate(long historicalMaximum, long current,
            List<Long> retries) {
        return Tensor2PerformanceGatePolicy.evaluate(
                Map.of("case", List.of(historicalMaximum - 10, historicalMaximum)),
                Map.of("case", current),
                Map.of("case", retries));
    }
}
