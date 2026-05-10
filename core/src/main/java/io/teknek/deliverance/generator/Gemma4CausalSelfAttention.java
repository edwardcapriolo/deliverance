package io.teknek.deliverance.generator;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.gemma4.Gemma4Config;
import io.teknek.deliverance.model.gemma4.Gemma4Model;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.VectorTensorMathUtils;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import net.jafama.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

public class Gemma4CausalSelfAttention extends CausalSelfAttention {
    private static final Logger logger = LoggerFactory.getLogger(Gemma4CausalSelfAttention.class);
    private static final boolean DEBUG_SHARED_KV = false;
    private static final boolean DISABLE_SHARED_KV = false;
    private final AbstractModel model;
    private final Gemma4Config config;
    private final int layerIndex;
    private final String layerType;
    private final boolean slidingAttention;
    private final boolean kvSharedLayer;
    private final boolean storeSharedKv;
    private final int headDim;
    private final int rotaryDim;
    private final int numberOfHeads;
    private final int numberOfKeyValueHeads;
    private final int numberOfKeyValueGroups;
    private final int queryLength;
    private final int kvLength;
    private final int slidingWindow;
    private final float[][] ropeFreqs;
    private final AbstractTensor queryWeights;
    private final AbstractTensor queryNormWeights;
    private final AbstractTensor outputProjectionWeights;
    private final Optional<AbstractTensor> keyWeights;
    private final Optional<AbstractTensor> valueWeights;
    private final Optional<AbstractTensor> keyNormWeights;
    private final ConfigurableTensorProvider configurableTensorProvider;
    private final MetricRegistry metricRegistry;

    public static boolean isSharedKvDisabled() {
        return DISABLE_SHARED_KV;
    }

    public Gemma4CausalSelfAttention(
            AbstractModel model,
            int layerIndex,
            String layerType,
            boolean kvSharedLayer,
            boolean storeSharedKv,
            AbstractTensor queryWeights,
            AbstractTensor queryNormWeights,
            Optional<AbstractTensor> keyWeights,
            Optional<AbstractTensor> valueWeights,
            Optional<AbstractTensor> keyNormWeights,
            AbstractTensor outputProjectionWeights,
            ConfigurableTensorProvider configurableTensorProvider,
            MetricRegistry metricRegistry
    ) {
        super(model, layerIndex, queryWeights, keyWeights.orElse(queryWeights), valueWeights.orElse(queryWeights),
                outputProjectionWeights, configurableTensorProvider, metricRegistry);
        this.model = model;
        this.config = (Gemma4Config) model.getConfig();
        this.layerIndex = layerIndex;
        this.layerType = layerType;
        this.slidingAttention = "sliding_attention".equals(layerType);
        this.kvSharedLayer = kvSharedLayer && !DISABLE_SHARED_KV;
        this.storeSharedKv = storeSharedKv && !DISABLE_SHARED_KV;
        this.headDim = config.getLayerHeadDim(layerType);
        this.rotaryDim = config.rotaryDimensionsByLayerType.get(layerType);
        this.numberOfHeads = config.numberOfHeads;
        this.numberOfKeyValueHeads = config.getLayerKeyValueProjectionHeads(layerType);
        this.numberOfKeyValueGroups = numberOfHeads / numberOfKeyValueHeads;
        this.queryLength = config.getLayerQueryProjectionLength(layerType);
        this.kvLength = config.getLayerKeyValueProjectionLength(layerType);
        this.slidingWindow = config.slidingWindow == null ? config.contextLength : config.slidingWindow;
        this.ropeFreqs = config.ropeFreqsByLayerType.get(layerType);
        this.queryWeights = queryWeights;
        this.queryNormWeights = queryNormWeights;
        this.keyWeights = keyWeights;
        this.valueWeights = valueWeights;
        this.keyNormWeights = keyNormWeights;
        this.outputProjectionWeights = outputProjectionWeights;
        this.configurableTensorProvider = configurableTensorProvider;
        this.metricRegistry = metricRegistry;
        logNormWeightSummary("q_norm_weight", queryNormWeights);
        keyNormWeights.ifPresent(w -> logNormWeightSummary("k_norm_weight", w));
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, int startPosition, KvBufferCache.KvBuffer kvMem,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        int batchSize = input.shape().first();
        metricRegistry.histogram("gemma4.attn.layer." + layerIndex + ".batch_size").update(batchSize);
        Timer.Context totalTimer = metricRegistry.timer("gemma4.attn.layer." + layerIndex + ".total").time();
        try (
                AbstractTensor queryBatch = model.makeDenseTensor(batchSize, queryLength);
                AbstractTensor keyBatch = model.makeDenseTensor(batchSize, kvLength);
                AbstractTensor valueBatch = model.makeDenseTensor(batchSize, kvLength);
                AbstractTensor valueOutput = model.makeDenseTensor(batchSize, queryLength)
        ) {
            Timer.Context projectTimer = metricRegistry.timer("gemma4.attn.layer." + layerIndex + ".project").time();
            try {
                project(input, queryBatch, queryWeights, queryLength);
                logFullAttentionSummary("q_after_proj", queryBatch);
                Gemma4RmsNormSupport.applyInPlace(queryBatch, numberOfHeads, headDim, config.layerNormEps, queryNormWeights);
                logFullAttentionSummary("q_after_norm", queryBatch);

                if (!kvSharedLayer) {
                    project(input, keyBatch, keyWeights.orElseThrow(), kvLength);
                    logFullAttentionSummary("k_after_proj", keyBatch);
                    project(input, valueBatch, valueWeights.orElse(keyWeights.orElseThrow()), kvLength);
                    logFullAttentionSummary("v_after_proj", valueBatch);
                    Gemma4RmsNormSupport.applyInPlace(keyBatch, numberOfKeyValueHeads, headDim, config.layerNormEps,
                            keyNormWeights.orElseThrow());
                    logFullAttentionSummary("k_after_norm", keyBatch);
                    Gemma4RmsNormSupport.applyInPlace(valueBatch, numberOfKeyValueHeads, headDim, config.layerNormEps, null);
                    logFullAttentionSummary("v_after_norm", valueBatch);
                }
            } finally {
                projectTimer.stop();
            }

            Timer.Context ropeTimer = metricRegistry.timer("gemma4.attn.layer." + layerIndex + ".rope_kv").time();
            try {
                applyRope(queryBatch, numberOfHeads, startPosition);
                if (!kvSharedLayer) {
                    applyRope(keyBatch, numberOfKeyValueHeads, startPosition);
                }
            } finally {
                ropeTimer.stop();
            }

            Timer.Context scoreTimer = metricRegistry.timer("gemma4.attn.layer." + layerIndex + ".score_value").time();
            try {
                int maxVisibleLength = slidingAttention
                        ? Math.min(slidingWindow, startPosition + batchSize)
                        : startPosition + batchSize;
                try (AbstractTensor packedKeys = model.getTensorAllocator().getDirty(keyBatch.dType(), TensorShape.of(maxVisibleLength, kvLength));
                     AbstractTensor packedValues = model.getTensorAllocator().getDirty(valueBatch.dType(), TensorShape.of(maxVisibleLength, kvLength));
                     AbstractTensor attn = model.makeDenseTensor(1, maxVisibleLength)) {
                    Gemma4Model.SharedKeyValues shared = kvSharedLayer ? ((Gemma4Model) model).getSharedKeyValues(layerType) : null;
                    if (DEBUG_SHARED_KV && (kvSharedLayer || storeSharedKv || "full_attention".equals(layerType))) {
                        String sharedShape = shared == null ? "none" : shared.key().shape() + "/" + shared.value().shape();
                        logger.debug("gemma4 attn_runtime layer={} type={} kvSharedLayer={} storeSharedKv={} queryLength={} kvLength={} sharedShape={}",
                                layerIndex, layerType, kvSharedLayer, storeSharedKv, queryLength, kvLength, sharedShape);
                    }
                    logFullAttentionSummary("q_after_rope", queryBatch);
                    if (kvSharedLayer && shared != null) {
                        logFullAttentionSummary("k_shared", shared.key());
                        logFullAttentionSummary("v_shared", shared.value());
                    } else {
                        logFullAttentionSummary("k_after_rope", keyBatch);
                        logFullAttentionSummary("v_after_norm_or_raw", valueBatch);
                    }
                    for (int position = startPosition, batchIndex = 0; position < startPosition + batchSize; position++, batchIndex++) {
                        AbstractTensor keyTensor = null;
                        AbstractTensor valueTensor = null;
                        AbstractTensor[] keyPages = null;
                        AbstractTensor[] valuePages = null;
                        if (!kvSharedLayer) {
                            keyTensor = kvMem.getKeyTensorForPosition(layerIndex, position);
                            valueTensor = kvMem.getValTensorForPosition(layerIndex, position);
                            logKvRowSummary("key_batch_before_cache", keyBatch, batchIndex, kvLength);
                            logKvRowSummary("value_batch_before_cache", valueBatch, batchIndex, kvLength);
                            copyKvRow(keyBatch, valueBatch, batchIndex, keyTensor, valueTensor);
                            logKvTensorSummary("key_cache_row", keyTensor, kvLength);
                            logKvTensorSummary("value_cache_row", valueTensor, kvLength);
                            keyPages = kvMem.getKeyTensorsUptoPosition(layerIndex, position);
                            valuePages = kvMem.getValTensorsUptoPosition(layerIndex, position);
                        }
                        int windowStart = slidingAttention ? Math.max(0, position - slidingWindow + 1) : 0;

                        try (AbstractTensor queryRow = queryBatch.slice(batchIndex);
                             AbstractTensor valueRow = valueOutput.slice(batchIndex)) {
                            int visibleLength = position - windowStart + 1;
                            if (kvSharedLayer) {
                                fillVisibleRowsFromDense(packedKeys, shared.key(), windowStart, visibleLength, kvLength);
                                fillVisibleRowsFromDense(packedValues, shared.value(), windowStart, visibleLength, kvLength);
                            } else {
                                fillVisibleRows(packedKeys, keyPages, position, windowStart, kvLength);
                                fillVisibleRows(packedValues, valuePages, position, windowStart, kvLength);
                            }
                            if (DEBUG_SHARED_KV && batchIndex == 0 && (kvSharedLayer || storeSharedKv || "full_attention".equals(layerType))) {
                                logger.debug("gemma4 attn_window layer={} type={} position={} windowStart={} visibleLength={} packedKeyShape={} packedValueShape={}",
                                        layerIndex,
                                        layerType,
                                        position,
                                        windowStart,
                                        visibleLength,
                                        packedKeys.shape(),
                                        packedValues.shape());
                            }
                            for (int head = 0; head < numberOfHeads; head++) {
                                int kvHead = head / numberOfKeyValueGroups;
                                int queryOffset = head * headDim;
                                int kvOffset = kvHead * headDim;

                                configurableTensorProvider.get().batchDotProduct(
                                        attn,
                                        queryRow,
                                        packedKeys,
                                        queryOffset,
                                        kvOffset,
                                        headDim,
                                        0,
                                        0,
                                        visibleLength
                                );
                                if (batchIndex == 0 && head == 0) {
                                    logFullAttentionSummary("attn_presoftcap", attn, visibleLength);
                                }
                                if (config.attnLogitSoftCapping != null) {
                                    for (int i = 0; i < visibleLength; i++) {
                                        float v = attn.get(0, i);
                                        v /= config.attnLogitSoftCapping;
                                        v = (float) FastMath.tanh(v);
                                        v *= config.attnLogitSoftCapping;
                                        attn.set(v, 0, i);
                                    }
                                }
                                if (batchIndex == 0 && head == 0) {
                                    logFullAttentionSummary("attn_postsoftcap", attn, visibleLength);
                                }
                                VectorTensorMathUtils.softMax(attn, 0, visibleLength);
                                if (batchIndex == 0 && head == 0) {
                                    logFullAttentionSummary("attn_postsoftmax", attn, visibleLength);
                                }

                                configurableTensorProvider.get().saxpy(
                                        attn,
                                        packedValues,
                                        valueRow,
                                        kvOffset,
                                        queryOffset,
                                        headDim,
                                        0,
                                        0,
                                        visibleLength
                                );
                                if (batchIndex == 0 && head == 0) {
                                    logFullAttentionSummary("value_output_partial", valueRow, queryLength);
                                }
                            }
                        }

                        if (!kvSharedLayer) {
                            keyTensor.close();
                            valueTensor.close();
                            closeAll(keyPages);
                            closeAll(valuePages);
                        }
                    }

                    if (storeSharedKv) {
                        int lastPosition = startPosition + batchSize - 1;
                        AbstractTensor[] keyPages = kvMem.getKeyTensorsUptoPosition(layerIndex, lastPosition);
                        AbstractTensor[] valuePages = kvMem.getValTensorsUptoPosition(layerIndex, lastPosition);
                        try (AbstractTensor fullKeys = model.getTensorAllocator().getDirty(keyBatch.dType(), TensorShape.of(lastPosition + 1, kvLength));
                             AbstractTensor fullValues = model.getTensorAllocator().getDirty(valueBatch.dType(), TensorShape.of(lastPosition + 1, kvLength))) {
                            fillVisibleRows(fullKeys, keyPages, lastPosition, 0, kvLength);
                            fillVisibleRows(fullValues, valuePages, lastPosition, 0, kvLength);
                            logKvTensorSummary("packed_full_k_row0", fullKeys, kvLength);
                            logKvTensorSummary("packed_full_v_row0", fullValues, kvLength);
                            logFullAttentionSummary("stored_full_k", fullKeys);
                            logFullAttentionSummary("stored_full_v", fullValues);
                            ((Gemma4Model) model).putSharedKeyValues(layerType, fullKeys, fullValues);
                        } finally {
                            closeAll(keyPages);
                            closeAll(valuePages);
                        }
                    }
                }
            } finally {
                scoreTimer.stop();
            }

            AbstractTensor result = model.makeDenseTensor(batchSize, config.embeddingLength);
            Timer.Context outProjTimer = metricRegistry.timer("gemma4.attn.layer." + layerIndex + ".out_proj").time();
            try (AbstractTensor valueQ = model.maybeQuantize(valueOutput)) {
                try {
                    configurableTensorProvider.get().dotProductChunk(result, valueQ, outputProjectionWeights, 0, queryLength, 0,
                            config.embeddingLength);
                    logFullAttentionSummary("o_proj", result);
                    tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(result)));
                } finally {
                    outProjTimer.stop();
                }
            }
            return result;
        } finally {
            totalTimer.stop();
        }
    }

    private void project(AbstractTensor input, AbstractTensor output, AbstractTensor weights, int outputLength) {
        configurableTensorProvider.get().dotProductChunk(output, input, weights, 0, config.embeddingLength, 0, outputLength);
    }

    /**
     * Packs the visible KV rows for one position into the front of a reusable dense tensor.
     *
     * This trades one copy per position for substantially fewer fragmented page walks per head,
     * which gives the tensor backend contiguous memory for dot products and SAXPY accumulation.
     * The destination tensor may be larger than the current visible window; only the first
     * `position - windowStart + 1` rows are populated.
     *
     * @return the number of rows written into {@code packed}
     */
    int fillVisibleRows(AbstractTensor packed, AbstractTensor[] pages, int position, int windowStart, int rowWidth) {
        int packedRow = 0;
        int globalOffset = 0;
        for (AbstractTensor page : pages) {
            int pageRows = Math.min(page.shape().first(), (position + 1) - globalOffset);
            int overlapStart = Math.max(windowStart, globalOffset);
            int overlapEnd = Math.min(position + 1, globalOffset + pageRows);
            if (overlapStart < overlapEnd) {
                int rowOffset = overlapStart - globalOffset;
                int size = overlapEnd - overlapStart;
                // Temporary debugging fallback: copy row-by-row until we prove the page-slice layout
                // matches the bulk copy assumptions here. If this fixes Gemma4 shared-KV reuse, this
                // should be revisited and removed in favor of a faster bulk path.
                for (int row = 0; row < size; row++) {
                    packed.copyFrom(page, page.getOffset(rowOffset + row, 0), packed.getOffset(packedRow, 0), rowWidth);
                    packedRow++;
                }
            }
            globalOffset += page.shape().first();
        }
        return packedRow;
    }

    /**
     * Shared-KV layers consume full-length same-type K/V states directly, so the visible window is
     * just a contiguous slice from the dense stored tensor.
     */
    int fillVisibleRowsFromDense(AbstractTensor packed, AbstractTensor dense, int windowStart, int visibleLength, int rowWidth) {
        packed.copyFrom(dense, dense.getOffset(windowStart, 0), packed.getOffset(0, 0), visibleLength * rowWidth);
        return visibleLength;
    }

    private void copyKvRow(AbstractTensor keyBatch, AbstractTensor valueBatch, int batchIndex, AbstractTensor keyTensor,
            AbstractTensor valueTensor) {
        try (AbstractTensor keyRow = keyBatch.slice(batchIndex); AbstractTensor valueRow = valueBatch.slice(batchIndex)) {
            if (keyTensor.dType() != keyBatch.dType()) {
                try (AbstractTensor keyQ = configurableTensorProvider.get().quantize(keyRow, keyTensor.dType(), 0, kvLength);
                     AbstractTensor valueQ = configurableTensorProvider.get().quantize(valueRow, valueTensor.dType(), 0, kvLength)) {
                    keyTensor.copyFrom(keyQ, 0, 0, kvLength);
                    valueTensor.copyFrom(valueQ, 0, 0, kvLength);
                }
            } else {
                keyTensor.copyFrom(keyRow, 0, 0, kvLength);
                valueTensor.copyFrom(valueRow, 0, 0, kvLength);
            }
        }
    }

    private void applyRope(AbstractTensor tensor, int headCount, int startPosition) {
        int halfRotaryDim = rotaryDim / 2;
        for (int batchIndex = 0; batchIndex < tensor.shape().first(); batchIndex++) {
            int position = startPosition + batchIndex;
            int freqOffset = position * halfRotaryDim;
            for (int head = 0; head < headCount; head++) {
                int offset = head * headDim;
                for (int i = 0; i < halfRotaryDim; i++) {
                    float first = tensor.get(batchIndex, offset + i);
                    float second = tensor.get(batchIndex, offset + i + halfRotaryDim);
                    float[] freq = ropeFreqs[freqOffset + i];
                    tensor.set(first * freq[0] - second * freq[1], batchIndex, offset + i);
                    tensor.set(first * freq[1] + second * freq[0], batchIndex, offset + i + halfRotaryDim);
                }
            }
        }
    }

    float score(AbstractTensor queryBatch, int batchIndex, int queryOffset, AbstractTensor keyPage, int row, int kvOffset) {
        float score = 0.0f;
        for (int i = 0; i < headDim; i++) {
            score += queryBatch.get(batchIndex, queryOffset + i) * keyPage.get(row, kvOffset + i);
        }
        if (config.attnLogitSoftCapping != null) {
            float scaled = score / config.attnLogitSoftCapping;
            return (float) (FastMath.tanh(scaled) * config.attnLogitSoftCapping);
        }
        return score;
    }

    void softmax(float[] scores) {
        float max = Float.NEGATIVE_INFINITY;
        for (float score : scores) {
            if (score > max) {
                max = score;
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < scores.length; i++) {
            scores[i] = (float) FastMath.exp(scores[i] - max);
            sum += scores[i];
        }
        for (int i = 0; i < scores.length; i++) {
            scores[i] /= sum;
        }
    }

    private void closeAll(AbstractTensor[] tensors) {
        for (AbstractTensor tensor : tensors) {
            tensor.close();
        }
    }

    private void logFullAttentionSummary(String stage, AbstractTensor tensor) {
        logFullAttentionSummary(stage, tensor, tensor.shape().last());
    }

    private void logFullAttentionSummary(String stage, AbstractTensor tensor, int width) {
        if (!DEBUG_SHARED_KV || (layerIndex != 4 && layerIndex != 14 && layerIndex != 34)) {
            return;
        }
        int effectiveWidth = Math.min(width, tensor.shape().last());
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        double sum = 0.0d;
        double sumSquares = 0.0d;
        StringBuilder first = new StringBuilder();
        int preview = Math.min(8, effectiveWidth);
        for (int i = 0; i < effectiveWidth; i++) {
            float v = tensor.get(0, i);
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            sumSquares += (double) v * v;
            if (i < preview) {
                if (i > 0) first.append(',');
                first.append(String.format(java.util.Locale.ROOT, "%.4f", v));
            }
        }
        double mean = sum / effectiveWidth;
        double l2 = Math.sqrt(sumSquares);
        logger.info("gemma4 attn_summary layer={} stage={} row0_min={} row0_max={} row0_mean={} row0_l2={} row0_first8=[{}]",
                layerIndex,
                stage,
                String.format(java.util.Locale.ROOT, "%.6f", min),
                String.format(java.util.Locale.ROOT, "%.6f", max),
                String.format(java.util.Locale.ROOT, "%.6f", mean),
                String.format(java.util.Locale.ROOT, "%.6f", l2),
                first);
    }

    private void logNormWeightSummary(String stage, AbstractTensor tensor) {
        if (!DEBUG_SHARED_KV || (layerIndex != 4 && layerIndex != 14 && layerIndex != 34)) {
            return;
        }
        int width = tensor.shape().dims() == 1 ? tensor.shape().first() : tensor.shape().last();
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        double sum = 0.0d;
        double sumSquares = 0.0d;
        StringBuilder first = new StringBuilder();
        int preview = Math.min(8, width);
        for (int i = 0; i < width; i++) {
            float v = tensor.shape().dims() == 1 ? tensor.get(i) : tensor.get(0, i);
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            sumSquares += (double) v * v;
            if (i < preview) {
                if (i > 0) first.append(',');
                first.append(String.format(java.util.Locale.ROOT, "%.4f", v));
            }
        }
        double mean = sum / width;
        double l2 = Math.sqrt(sumSquares);
        logger.info("gemma4 attn_weight_summary layer={} stage={} min={} max={} mean={} l2={} first8=[{}]",
                layerIndex,
                stage,
                String.format(java.util.Locale.ROOT, "%.6f", min),
                String.format(java.util.Locale.ROOT, "%.6f", max),
                String.format(java.util.Locale.ROOT, "%.6f", mean),
                String.format(java.util.Locale.ROOT, "%.6f", l2),
                first);
    }

    private void logKvRowSummary(String stage, AbstractTensor tensor, int batchIndex, int width) {
        if (!DEBUG_SHARED_KV || layerIndex != 14 || batchIndex != 0) {
            return;
        }
        try (AbstractTensor row = tensor.slice(batchIndex)) {
            logKvTensorSummary(stage, row, width);
        }
    }

    private void logKvTensorSummary(String stage, AbstractTensor tensor, int width) {
        if (!DEBUG_SHARED_KV || layerIndex != 14) {
            return;
        }
        int effectiveWidth = Math.min(width, tensor.shape().last());
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        double sum = 0.0d;
        double sumSquares = 0.0d;
        StringBuilder first = new StringBuilder();
        int preview = Math.min(8, effectiveWidth);
        for (int i = 0; i < effectiveWidth; i++) {
            float v = tensor.get(0, i);
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            sumSquares += (double) v * v;
            if (i < preview) {
                if (i > 0) first.append(',');
                first.append(String.format(java.util.Locale.ROOT, "%.4f", v));
            }
        }
        double mean = sum / effectiveWidth;
        double l2 = Math.sqrt(sumSquares);
        logger.info("gemma4 kv_summary layer={} stage={} row0_min={} row0_max={} row0_mean={} row0_l2={} row0_first8=[{}]",
                layerIndex,
                stage,
                String.format(java.util.Locale.ROOT, "%.6f", min),
                String.format(java.util.Locale.ROOT, "%.6f", max),
                String.format(java.util.Locale.ROOT, "%.6f", mean),
                String.format(java.util.Locale.ROOT, "%.6f", l2),
                first);
    }
}
