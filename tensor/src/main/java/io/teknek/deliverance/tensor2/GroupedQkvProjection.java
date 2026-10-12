package io.teknek.deliverance.tensor2;

import java.util.Objects;

/** Executes one grouped Q/K/V output-row chunk using TensorRef provider operations. */
public record GroupedQkvProjection(TensorRef input, TensorRef query, TensorRef key, TensorRef value,
        TensorRef queryWeights, TensorRef keyWeights, TensorRef valueWeights, int inputLength, int offset,
        int queryLength, int kvLength) {
    public GroupedQkvProjection {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(queryWeights, "queryWeights");
        Objects.requireNonNull(keyWeights, "keyWeights");
        Objects.requireNonNull(valueWeights, "valueWeights");
        if (offset < 0 || queryLength < 0 || kvLength < 0 || inputLength <= 0) {
            throw new IllegalArgumentException("invalid grouped QKV projection range");
        }
    }

    public void execute(Lighter lighter) {
        lighter.dotProductRows(query, input, queryWeights, 0, inputLength, offset, queryLength, offset);
        int keyValueLength = Math.min(queryLength, kvLength - offset);
        if (keyValueLength <= 0) {
            return;
        }
        lighter.dotProductRows(key, input, keyWeights, 0, inputLength, offset, keyValueLength, offset);
        lighter.dotProductRows(value, input, valueWeights, 0, inputLength, offset, keyValueLength, offset);
    }
}
