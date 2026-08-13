package io.teknek.deliverance.model.bert;

import java.util.Arrays;

/**
 * Hugging Face BERT model inputs flattened from {@code [batch, seq]} to row-major arrays.
 *
 * <p>HF defaults are represented explicitly: absent {@code token_type_ids} become zeros, and absent
 * {@code position_ids} become {@code past_key_values_length..past_key_values_length + seq - 1}, broadcast
 * to every batch row.</p>
 */
public final class BertInput {
    private final int[] inputIds;
    private final int[] attentionMask;
    private final int[] tokenTypeIds;
    private final int[] positionIds;
    private final int batchSize;
    private final int sequenceLength;

    public BertInput(int[] inputIds, int[] attentionMask, int[] tokenTypeIds, int[] positionIds,
            int batchSize, int sequenceLength) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive");
        }
        if (sequenceLength <= 0) {
            throw new IllegalArgumentException("sequenceLength must be positive");
        }
        int flattenedLength = batchSize * sequenceLength;
        this.inputIds = copyRequired(inputIds, flattenedLength, "input_ids");
        this.attentionMask = copyOrDefault(attentionMask, flattenedLength, 1, "attention_mask");
        this.tokenTypeIds = copyOrDefault(tokenTypeIds, flattenedLength, 0, "token_type_ids");
        this.positionIds = positionIds == null
                ? defaultPositionIds(batchSize, sequenceLength, 0)
                : copyRequired(positionIds, flattenedLength, "position_ids");
        this.batchSize = batchSize;
        this.sequenceLength = sequenceLength;
    }

    public static BertInput singleSequence(int[] inputIds) {
        return singleSequence(inputIds, null, null, null);
    }

    public static BertInput singleSequence(int[] inputIds, int[] attentionMask, int[] tokenTypeIds, int[] positionIds) {
        return new BertInput(inputIds, attentionMask, tokenTypeIds, positionIds, 1, inputIds.length);
    }

    public static BertInput withPastKeyValuesLength(int[] inputIds, int batchSize, int sequenceLength,
            int pastKeyValuesLength) {
        return new BertInput(inputIds, null, null,
                defaultPositionIds(batchSize, sequenceLength, pastKeyValuesLength), batchSize, sequenceLength);
    }

    public int[] inputIds() {
        return inputIds;
    }

    public int[] attentionMask() {
        return attentionMask;
    }

    public int[] tokenTypeIds() {
        return tokenTypeIds;
    }

    public int[] positionIds() {
        return positionIds;
    }

    public int batchSize() {
        return batchSize;
    }

    public int sequenceLength() {
        return sequenceLength;
    }

    public int flattenedLength() {
        return inputIds.length;
    }

    private static int[] copyRequired(int[] values, int expectedLength, String name) {
        if (values == null) {
            throw new IllegalArgumentException(name + " is required");
        }
        if (values.length != expectedLength) {
            throw new IllegalArgumentException(name + " length " + values.length + " != " + expectedLength);
        }
        return Arrays.copyOf(values, values.length);
    }

    private static int[] copyOrDefault(int[] values, int expectedLength, int defaultValue, String name) {
        if (values != null) {
            return copyRequired(values, expectedLength, name);
        }
        int[] defaults = new int[expectedLength];
        Arrays.fill(defaults, defaultValue);
        return defaults;
    }

    private static int[] defaultPositionIds(int batchSize, int sequenceLength, int pastKeyValuesLength) {
        int[] positions = new int[batchSize * sequenceLength];
        for (int batch = 0; batch < batchSize; batch++) {
            int rowOffset = batch * sequenceLength;
            for (int token = 0; token < sequenceLength; token++) {
                positions[rowOffset + token] = pastKeyValuesLength + token;
            }
        }
        return positions;
    }
}
