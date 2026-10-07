package io.teknek.deliverance.generator2;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.kv.KvReadView;
import io.teknek.deliverance.tensor2.BatchDotProduct;
import io.teknek.deliverance.tensor2.ScaledSoftMax;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Optional;

/** TensorRef port of {@code PackedBlockAttention}; keep control flow mechanically aligned. */
public final class PackedBlockAttention2 {
    private static final long MAX_CONCURRENT_SCORE_BYTES = 16L * 1024L * 1024L;
    private static final int MAX_SCORE_ELEMENTS_PER_HEAD_TASK = 128 * 1024;

    private final AbstractModel model;
    private final MetricRegistry metricRegistry;
    private final int layerIndex;

    public PackedBlockAttention2(AbstractModel model, MetricRegistry metricRegistry) {
        this(model, metricRegistry, -1);
    }

    public PackedBlockAttention2(AbstractModel model, MetricRegistry metricRegistry, int layerIndex) {
        this.model = java.util.Objects.requireNonNull(model, "model");
        this.metricRegistry = java.util.Objects.requireNonNull(metricRegistry, "metricRegistry");
        this.layerIndex = layerIndex;
    }

    public void forward(TensorRef output, TensorRef query, TensorRef keys, TensorRef values,
            int prefixRows, int queryRows, int numberOfHeads, int numberOfKeyValueHeads, int headSize,
            float scale, Float softcap, boolean causalWithinBlock) {
        Preconditions.checkArgument(query.shape().first() == queryRows, "query rows mismatch");
        Preconditions.checkArgument(output.shape().first() == queryRows, "output rows mismatch");
        Preconditions.checkArgument(keys.shape().first() >= prefixRows + queryRows, "keys missing visible rows");
        Preconditions.checkArgument(values.shape().first() >= prefixRows + queryRows, "values missing visible rows");
        Preconditions.checkArgument(numberOfHeads % numberOfKeyValueHeads == 0, "GQA heads must divide evenly");
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        output.memorySegment().fill((byte) 0);
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "packedblockattention2.score_value").time()) {
            int totalVisibleRows = prefixRows + queryRows;
            int queryBlockRows = queryBlockRows(queryRows, totalVisibleRows);
            int headSplits = headSplitCount(numberOfHeads, queryBlockRows, totalVisibleRows);
            model.runChunks("packedblockattention2.heads", 0, numberOfHeads, headSplits,
                    Optional.empty(), (headStart, headCount) -> {
                int headEnd = headStart + headCount;
                for (int head = headStart; head < headEnd; head++) {
                    int kvHead = head / headGroupSize;
                    int queryOffset = head * headSize;
                    int kvOffset = kvHead * headSize;
                    for (int queryRowStart = 0; queryRowStart < queryRows; queryRowStart += queryBlockRows) {
                        int blockRows = Math.min(queryBlockRows, queryRows - queryRowStart);
                        try (TensorRef scores = model.makeDenseTensorRef(blockRows, totalVisibleRows)) {
                            model.getLighter().batchDotProduct(new BatchDotProduct()
                                    .result(scores)
                                    .a(query)
                                    .b(keys)
                                    .aRowOffset(queryRowStart)
                                    .aColumnOffset(queryOffset)
                                    .bColumnOffset(kvOffset)
                                    .columnLength(headSize)
                                    .resultRowOffset(0)
                                    .bRowOffset(0)
                                    .rowChunkSize(totalVisibleRows));
                            for (int localRow = 0; localRow < blockRows; localRow++) {
                                int row = queryRowStart + localRow;
                                int visibleRows = causalWithinBlock ? prefixRows + row + 1 : totalVisibleRows;
                                try (TensorRef scoreRow = scores.slice(localRow);
                                     TensorRef outputRow = output.slice(row)) {
                                    if (prefixRows + row == totalVisibleRows - 1) {
                                        emitAttentionDebug("attention_scores", head, prefixRows + row, scoreRow);
                                    }
                                    model.getCompositeOps().scaledSoftMax(new ScaledSoftMax(scale)
                                            .target(scoreRow)
                                            .offsetAndLength(0, visibleRows)
                                            .softcap(softcap));
                                    if (prefixRows + row == totalVisibleRows - 1) {
                                        emitAttentionDebug("attention_probabilities", head, prefixRows + row, scoreRow);
                                    }
                                    model.getLighter().saxpy(scoreRow, values, outputRow, kvOffset, queryOffset,
                                            headSize, 0, 0, visibleRows);
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    public void forward(TensorRef output, TensorRef query, KvReadView prefixView, TensorRef currentKeys,
            TensorRef currentValues, int prefixRows, int queryRows, int numberOfHeads, int numberOfKeyValueHeads,
            int headSize, float scale, Float softcap, boolean causalWithinBlock) {
        Preconditions.checkArgument(query.shape().first() == queryRows, "query rows mismatch");
        Preconditions.checkArgument(output.shape().first() == queryRows, "output rows mismatch");
        Preconditions.checkArgument(prefixView.visibleTokens() == prefixRows, "prefix rows mismatch");
        Preconditions.checkArgument(currentKeys.shape().first() == queryRows, "current key rows mismatch");
        Preconditions.checkArgument(currentValues.shape().first() == queryRows, "current value rows mismatch");
        Preconditions.checkArgument(numberOfHeads % numberOfKeyValueHeads == 0, "GQA heads must divide evenly");
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        output.memorySegment().fill((byte) 0);
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "packedblockattention2.score_value").time()) {
            for (int row = 0; row < queryRows; row++) {
                int visibleRows = causalWithinBlock ? prefixRows + row + 1 : prefixRows + queryRows;
                try (TensorRef queryRow = query.slice(row);
                     TensorRef outputRow = output.slice(row);
                     TensorRef scores = model.makeDenseTensorRef(1, visibleRows)) {
                    for (int head = 0; head < numberOfHeads; head++) {
                        int kvHead = head / headGroupSize;
                        int queryOffset = head * headSize;
                        int kvOffset = kvHead * headSize;
                        scoreRows(scores, queryRow, prefixView, currentKeys, prefixRows, visibleRows,
                                queryOffset, kvOffset, headSize);
                        model.getCompositeOps().scaledSoftMax(new ScaledSoftMax(scale)
                                .target(scores)
                                .offsetAndLength(0, visibleRows)
                                .softcap(softcap));
                        accumulateRows(scores, outputRow, prefixView, currentValues, prefixRows, visibleRows,
                                kvOffset, queryOffset, headSize);
                    }
                }
            }
        }
    }

    public void decodePagedAttention(TensorRef output, TensorRef query, TensorRef[] keyPages, TensorRef[] valuePages,
            int visibleRows, int numberOfHeads, int numberOfKeyValueHeads, int headSize, float scale, Float softcap) {
        Preconditions.checkArgument(keyPages.length == valuePages.length, "key/value page count mismatch");
        Preconditions.checkArgument(query.shape().first() == 1, "decode query must have one row");
        Preconditions.checkArgument(output.shape().first() == 1, "decode output must have one row");
        Preconditions.checkArgument(numberOfHeads % numberOfKeyValueHeads == 0, "GQA heads must divide evenly");
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        output.memorySegment().fill((byte) 0);
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "packedblockattention2.decode_paged_attention").time()) {
            model.runChunks("packedblockattention2.decode_heads", 0, numberOfHeads,
                    Math.max(1, Math.min(numberOfHeads, model.getPool().getCoreCount())), Optional.empty(),
                    (headStart, headCount) -> {
                int headEnd = headStart + headCount;
                for (int head = headStart; head < headEnd; head++) {
                    int kvHead = head / headGroupSize;
                    int queryOffset = head * headSize;
                    int kvOffset = kvHead * headSize;
                    try (TensorRef scores = model.makeDenseTensorRef(1, visibleRows)) {
                        int globalRow = 0;
                        for (int pageIndex = 0; pageIndex < keyPages.length; pageIndex++) {
                            if (globalRow >= visibleRows) {
                                break;
                            }
                            TensorRef keyPage = keyPages[pageIndex];
                            int rows = pageRows(keyPage, pageIndex, keyPages.length, globalRow, visibleRows);
                            if (rows <= 0) {
                                continue;
                            }
                            model.getLighter().batchDotProduct(new BatchDotProduct()
                                    .result(scores)
                                    .a(query)
                                    .b(keyPage)
                                    .aColumnOffset(queryOffset)
                                    .bColumnOffset(kvOffset)
                                    .columnLength(headSize)
                                    .resultRowOffset(globalRow)
                                    .bRowOffset(0)
                                    .rowChunkSize(rows));
                            globalRow += rows;
                        }
                        emitAttentionDebug("attention_scores", head, visibleRows - 1, scores);
                        model.getCompositeOps().scaledSoftMax(new ScaledSoftMax(scale)
                                .target(scores)
                                .offsetAndLength(0, visibleRows)
                                .softcap(softcap));
                        emitAttentionDebug("attention_probabilities", head, visibleRows - 1, scores);
                        globalRow = 0;
                        try (TensorRef outputRow = output.slice(0)) {
                            for (int pageIndex = 0; pageIndex < valuePages.length; pageIndex++) {
                                if (globalRow >= visibleRows) {
                                    break;
                                }
                                TensorRef valuePage = valuePages[pageIndex];
                                int rows = pageRows(valuePage, pageIndex, valuePages.length, globalRow, visibleRows);
                                if (rows <= 0) {
                                    continue;
                                }
                                model.getLighter().saxpy(scores, valuePage, outputRow, kvOffset, queryOffset,
                                        headSize, globalRow, 0, rows);
                                globalRow += rows;
                            }
                        }
                    }
                }
            });
        }
    }

    private int pageRows(TensorRef page, int pageIndex, int pageCount, int globalRow, int visibleRows) {
        if (pageIndex == pageCount - 1) {
            return Math.min(page.shape().first(), visibleRows - globalRow);
        }
        int prefixRows = visibleRows - 1;
        return Math.min(page.shape().first(), prefixRows - globalRow);
    }

    private void scoreRows(TensorRef scores, TensorRef queryRow, KvReadView prefixView, TensorRef currentKeys,
            int prefixRows, int visibleRows, int queryOffset, int kvOffset, int headSize) {
        for (int keyPosition = 0; keyPosition < visibleRows; keyPosition++) {
            try (TensorRef keyRow = keyPosition < prefixRows
                    ? prefixView.keyRowRef(keyPosition)
                    : currentKeys.slice(keyPosition - prefixRows)) {
                model.getLighter().dotProductRows(scores, queryRow, keyRow, queryOffset, kvOffset, headSize,
                        0, 1, keyPosition);
            }
        }
    }

    private void accumulateRows(TensorRef scores, TensorRef outputRow, KvReadView prefixView,
            TensorRef currentValues, int prefixRows, int visibleRows, int kvOffset, int queryOffset, int headSize) {
        for (int valuePosition = 0; valuePosition < visibleRows; valuePosition++) {
            try (TensorRef valueRow = valuePosition < prefixRows
                    ? prefixView.valueRowRef(valuePosition)
                    : currentValues.slice(valuePosition - prefixRows)) {
                model.getLighter().saxpy(scores.get(0, valuePosition), valueRow, outputRow,
                        kvOffset, queryOffset, headSize);
            }
        }
    }

    private void emitAttentionDebug(String stage, int head, int position, TensorRef tensor) {
        if (layerIndex >= 0) {
            model.emitLayerDebug(layerIndex, stage + ".head" + head + ".position" + position, tensor);
        }
    }

    private int queryBlockRows(int queryRows, int visibleRows) {
        return Math.max(1, Math.min(queryRows, Math.max(1, MAX_SCORE_ELEMENTS_PER_HEAD_TASK / visibleRows)));
    }

    private int headSplitCount(int numberOfHeads, int queryBlockRows, int visibleRows) {
        long scoreBytes = (long) queryBlockRows * visibleRows * Float.BYTES;
        if (scoreBytes <= 0) {
            return 1;
        }
        long memoryBound = Math.max(1L, MAX_CONCURRENT_SCORE_BYTES / scoreBytes);
        int poolBound = Math.max(1, model.getPool().getCoreCount());
        return (int) Math.max(1L, Math.min(Math.min(numberOfHeads, poolBound), memoryBound));
    }
}
