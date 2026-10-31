package io.teknek.deliverance.model.gemma4;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Characterizes V2 shared-KV history independently of model and generation code. */
class Gemma4V2SharedKvStateCharacterizationTest {

    @Test
    void givenOrdinaryKvHasRestoredPrefix_whenSharedKvReceivesSuffix_thenSharedKvRetainsPrefix() {
        Lighter lighter = new Lighter(new MetricRegistry());
        int restoredPrefixRows = 4;
        int suffixRows = 1;
        try (Gemma4Model.V2SharedKeyValues state = new Gemma4Model.V2SharedKeyValues();
             TensorRef prefixKey = rows(lighter, restoredPrefixRows, 10.0f);
             TensorRef prefixValue = rows(lighter, restoredPrefixRows, 110.0f);
             TensorRef suffixKey = rows(lighter, 1, 40.0f);
             TensorRef suffixValue = rows(lighter, 1, 140.0f)) {
            // Given: ordinary KV and shared KV both contain the restored four-row prefix.
            state.store(0, 0, prefixKey, prefixValue, lighter);
            // When: the shared state receives the one-row suffix through its current API.
            whenSharedKvReceivesSuffix(state, restoredPrefixRows, suffixKey, suffixValue, lighter);

            try (TensorRef key = state.get(0, true); TensorRef value = state.get(0, false)) {
                // Then: shared KV must contain the restored prefix plus the suffix.
                assertEquals(restoredPrefixRows + suffixRows, key.shape().first());
                assertEquals(restoredPrefixRows + suffixRows, value.shape().first());
                assertEquals(40.0f, key.get(4, 0));
                assertEquals(140.0f, value.get(4, 0));
            }
        }
    }

    @Test
    void givenSharedKvHasFourRows_whenOrdinaryKvReplaysAtPositionTwo_thenSharedKvHasThreeRows() {
        Lighter lighter = new Lighter(new MetricRegistry());
        int replayPosition = 2;
        try (Gemma4Model.V2SharedKeyValues state = new Gemma4Model.V2SharedKeyValues();
             TensorRef originalKey = rows(lighter, 4, 10.0f);
             TensorRef originalValue = rows(lighter, 4, 110.0f);
             TensorRef replayKey = rows(lighter, 1, 50.0f);
             TensorRef replayValue = rows(lighter, 1, 150.0f)) {
            // Given: both caches initially represent four rows.
            state.store(0, 0, originalKey, originalValue, lighter);

            // When: ordinary KV replays at position two, its history becomes rows [0, 2), then row two is written.
            whenOrdinaryKvReplaysAtPosition(state, replayPosition, replayKey, replayValue, lighter);

            try (TensorRef key = state.get(0, true); TensorRef value = state.get(0, false)) {
                // Then: shared KV must represent rows [0, 3), not append after the old four rows.
                assertEquals(replayPosition + 1, key.shape().first());
                assertEquals(replayPosition + 1, value.shape().first());
                assertEquals(50.0f, key.get(2, 0));
                assertEquals(150.0f, value.get(2, 0));
            }
        }
    }

    private static void whenSharedKvReceivesSuffix(Gemma4Model.V2SharedKeyValues state, int startPosition,
            TensorRef suffixKey, TensorRef suffixValue, Lighter lighter) {
        state.store(0, startPosition, suffixKey, suffixValue, lighter);
    }

    private static void whenOrdinaryKvReplaysAtPosition(Gemma4Model.V2SharedKeyValues state, int replayPosition,
            TensorRef replayKey, TensorRef replayValue, Lighter lighter) {
        state.store(0, replayPosition, replayKey, replayValue, lighter);
    }

    private static TensorRef rows(Lighter lighter, int count, float firstValue) {
        TensorRef result = lighter.allocate(DType.F32, TensorShape.of(count, 1));
        for (int row = 0; row < count; row++) {
            result.set(firstValue + row, row, 0);
        }
        return result;
    }
}
