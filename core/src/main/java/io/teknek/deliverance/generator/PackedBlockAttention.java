package io.teknek.deliverance.generator;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.kv.KvReadView;
import io.teknek.deliverance.tensor.operations.TensorOperations;

/** Packed prefix-plus-block attention composed from primitive tensor operations. */
final class PackedBlockAttention {
    private final AbstractModel model;
    private final MetricRegistry metricRegistry;

    PackedBlockAttention(AbstractModel model, MetricRegistry metricRegistry) {
        this.model = model;
        this.metricRegistry = metricRegistry;
    }

    void forward(AbstractTensor output, AbstractTensor query, AbstractTensor keys, AbstractTensor values,
            int prefixRows, int queryRows, int numberOfHeads, int numberOfKeyValueHeads, int headSize,
            float scale, Float softcap, boolean causalWithinBlock) {
        Preconditions.checkArgument(query.shape().first() == queryRows, "query rows mismatch");
        Preconditions.checkArgument(output.shape().first() == queryRows, "output rows mismatch");
        Preconditions.checkArgument(keys.shape().first() >= prefixRows + queryRows, "keys missing visible rows");
        Preconditions.checkArgument(values.shape().first() >= prefixRows + queryRows, "values missing visible rows");
        Preconditions.checkArgument(numberOfHeads % numberOfKeyValueHeads == 0, "GQA heads must divide evenly");
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        TensorOperations ops = model.primaryTensorOperations();
        output.clear();
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "packedblockattention.score_value").time()) {
            for (int row = 0; row < queryRows; row++) {
                int visibleRows = causalWithinBlock ? prefixRows + row + 1 : prefixRows + queryRows;
                try (AbstractTensor queryRow = query.slice(row);
                     AbstractTensor outputRow = output.slice(row);
                     AbstractTensor scores = model.makeDenseTensor(1, visibleRows)) {
                    for (int head = 0; head < numberOfHeads; head++) {
                        int kvHead = head / headGroupSize;
                        int queryOffset = head * headSize;
                        int kvOffset = kvHead * headSize;
                        ops.batchDotProduct(scores, queryRow, keys, queryOffset, kvOffset, headSize, 0, 0,
                                visibleRows);
                        ops.scaledSoftMax(scores, 0, visibleRows, scale, softcap);
                        ops.saxpy(scores, values, outputRow, kvOffset, queryOffset, headSize, 0, 0,
                                visibleRows);
                    }
                }
            }
        }
    }

    void forward(AbstractTensor output, AbstractTensor query, KvReadView prefixView, AbstractTensor currentKeys,
            AbstractTensor currentValues, int prefixRows, int queryRows, int numberOfHeads, int numberOfKeyValueHeads,
            int headSize, float scale, Float softcap, boolean causalWithinBlock) {
        Preconditions.checkArgument(query.shape().first() == queryRows, "query rows mismatch");
        Preconditions.checkArgument(output.shape().first() == queryRows, "output rows mismatch");
        Preconditions.checkArgument(prefixView.visibleTokens() == prefixRows, "prefix rows mismatch");
        Preconditions.checkArgument(currentKeys.shape().first() == queryRows, "current key rows mismatch");
        Preconditions.checkArgument(currentValues.shape().first() == queryRows, "current value rows mismatch");
        Preconditions.checkArgument(numberOfHeads % numberOfKeyValueHeads == 0, "GQA heads must divide evenly");
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        TensorOperations ops = model.primaryTensorOperations();
        output.clear();
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "packedblockattention.score_value").time()) {
            for (int row = 0; row < queryRows; row++) {
                int visibleRows = causalWithinBlock ? prefixRows + row + 1 : prefixRows + queryRows;
                try (AbstractTensor queryRow = query.slice(row);
                     AbstractTensor outputRow = output.slice(row);
                     AbstractTensor scores = model.makeDenseTensor(1, visibleRows)) {
                    for (int head = 0; head < numberOfHeads; head++) {
                        int kvHead = head / headGroupSize;
                        int queryOffset = head * headSize;
                        int kvOffset = kvHead * headSize;
                        scoreRows(ops, scores, queryRow, prefixView, currentKeys, prefixRows, visibleRows,
                                queryOffset, kvOffset, headSize);
                        ops.scaledSoftMax(scores, 0, visibleRows, scale, softcap);
                        accumulateRows(ops, scores, outputRow, prefixView, currentValues, prefixRows, visibleRows,
                                kvOffset, queryOffset, headSize);
                    }
                }
            }
        }
    }

    private void scoreRows(TensorOperations ops, AbstractTensor scores, AbstractTensor queryRow, KvReadView prefixView,
            AbstractTensor currentKeys, int prefixRows, int visibleRows, int queryOffset, int kvOffset, int headSize) {
        for (int keyPosition = 0; keyPosition < visibleRows; keyPosition++) {
            try (AbstractTensor keyRow = keyPosition < prefixRows
                    ? prefixView.keyRow(keyPosition)
                    : currentKeys.slice(keyPosition - prefixRows)) {
                scores.set(ops.dotProduct(queryRow, keyRow, queryOffset, kvOffset, headSize), 0, keyPosition);
            }
        }
    }

    private void accumulateRows(TensorOperations ops, AbstractTensor scores, AbstractTensor outputRow,
            KvReadView prefixView, AbstractTensor currentValues, int prefixRows, int visibleRows, int kvOffset,
            int queryOffset, int headSize) {
        for (int valuePosition = 0; valuePosition < visibleRows; valuePosition++) {
            try (AbstractTensor valueRow = valuePosition < prefixRows
                    ? prefixView.valueRow(valuePosition)
                    : currentValues.slice(valuePosition - prefixRows)) {
                ops.saxpy(scores.get(0, valuePosition), valueRow, outputRow, kvOffset, queryOffset, headSize);
            }
        }
    }
}
