package io.teknek.deliverance.generator2;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.kv.CacheExecutionMode;
import io.teknek.deliverance.tensor.kv.AttentionPattern;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor.kv.KvReadView;
import io.teknek.deliverance.tensor.kv.KvWriteCursor;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.generator.ForwardPhase;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** TensorRef-native base for self-attention backed by KV cache v2. */
public abstract class KvCacheSelfAttention2 extends BaseCausalSelfAttention2 {
    protected final AbstractModel model;
    protected final int layerIndex;
    protected final io.teknek.deliverance.safetensors.Config config;
    protected final TensorRef queryAttentionWeights;
    protected final TensorRef keyAttentionWeights;
    protected final TensorRef valueAttentionWeights;
    protected final TensorRef outputProjectionWeights;
    protected final MetricRegistry metricRegistry;
    protected final String queryWeightName;
    protected final String keyWeightName;
    protected final String valueWeightName;
    protected final String outputWeightName;
    protected final int attentionLength;
    protected final int kvLength;
    protected final int numberOfHeads;
    protected final int numberOfKeyValueHeads;
    protected final int headGroupSize;
    protected final float attentionScale;
    protected final Lighter lighter;

    protected KvCacheSelfAttention2(AbstractModel model, int layerIndex, TensorRef queryAttentionWeights,
            TensorRef keyAttentionWeights, TensorRef valueAttentionWeights, TensorRef outputProjectionWeights,
            Lighter lighter, MetricRegistry metricRegistry, String queryWeightName, String keyWeightName,
            String valueWeightName, String outputWeightName) {
        super(lighter);
        this.model = java.util.Objects.requireNonNull(model, "model");
        this.layerIndex = layerIndex;
        this.config = model.getConfig();
        this.queryAttentionWeights = java.util.Objects.requireNonNull(queryAttentionWeights, "queryAttentionWeights");
        this.keyAttentionWeights = java.util.Objects.requireNonNull(keyAttentionWeights, "keyAttentionWeights");
        this.valueAttentionWeights = java.util.Objects.requireNonNull(valueAttentionWeights, "valueAttentionWeights");
        this.outputProjectionWeights = java.util.Objects.requireNonNull(outputProjectionWeights,
                "outputProjectionWeights");
        this.lighter = java.util.Objects.requireNonNull(lighter, "lighter");
        this.metricRegistry = java.util.Objects.requireNonNull(metricRegistry, "metricRegistry");
        this.queryWeightName = queryWeightName;
        this.keyWeightName = keyWeightName;
        this.valueWeightName = valueWeightName;
        this.outputWeightName = outputWeightName;
        this.attentionLength = model.getLocalAttentionLength();
        this.kvLength = model.getLocalKvLength();
        this.numberOfHeads = model.getLocalNumberOfHeads();
        this.numberOfKeyValueHeads = model.getLocalNumberOfKeyValueHeads();
        Preconditions.checkArgument(numberOfKeyValueHeads > 0 && numberOfHeads % numberOfKeyValueHeads == 0,
                "Attention heads must be divisible by KV heads");
        this.headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        this.attentionScale = config.attentionMultiplier != null
                ? config.attentionMultiplier
                : (float) (1.0 / StrictMath.sqrt(config.headSize));
    }

    @Override
    public TensorRef forward(TensorRef input, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase) {
        Preconditions.checkArgument(input.dims() == 2 && input.shape().last() == config.embeddingLength,
                "Attention input must be [batch, embeddingLength]");
        Preconditions.checkArgument(startPosition >= 0 && startPosition <= kvSession.length(),
                "startPosition must be within KV session length");
        CacheExecutionMode mode = phase == ForwardPhase.DECODE
                ? CacheExecutionMode.DECODE_UPDATE_CACHE
                : CacheExecutionMode.PREFILL_UPDATE_CACHE;
        int batchSize = input.shape().first();
        Preconditions.checkArgument(mode != CacheExecutionMode.DECODE_UPDATE_CACHE || batchSize == 1,
                "decode update expects one token at a time");

        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "kvcacheselfattention2.forward").time()) {
            TensorRef[] qkv = projectQkv(input, phase);
            try {
                normalizeQueryKey(qkv[0], qkv[1]);
                applyRotaryEmbedding(qkv[0], qkv[1], startPosition);
                if (writesCache(mode)) {
                    writeKvRows(kvSession, mode, qkv[1], qkv[2], startPosition);
                }
                TensorRef attended = attend(qkv[0], qkv[1], qkv[2], kvSession, startPosition, mode,
                        tensorReducer, phase);
                try {
                    return projectOutput(attended, phase);
                } finally {
                    attended.close();
                }
            } finally {
                closeAll(qkv);
            }
        }
    }

    protected void normalizeQueryKey(TensorRef query, TensorRef key) {
    }

    protected void applyRotaryEmbedding(TensorRef query, TensorRef key, int startPosition) {
    }

    protected final TensorRef attend(TensorRef query, TensorRef currentKeys, TensorRef currentValues,
            KvCacheSession kvSession, int startPosition, CacheExecutionMode mode,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase) {
        int batchSize = query.shape().first();
        int capacity = startPosition + batchSize;
        TensorRef packedKeys = model.makeDenseTensorRef(capacity, kvLength);
        TensorRef packedValues = model.makeDenseTensorRef(capacity, kvLength);
        TensorRef scores = model.makeDenseTensorRef(batchSize, capacity);
        TensorRef attended = model.makeDenseTensorRef(batchSize, attentionLength);
        attended.memorySegment().fill((byte) 0);
        try (KvReadView readView = kvSession.readView(layerIndex, startPosition, AttentionPattern.CAUSAL)) {
            TensorRef prefixKeys = readView.copyVisibleKeysRef();
            TensorRef prefixValues = readView.copyVisibleValuesRef();
            try {
                for (int row = 0; row < startPosition; row++) {
                    copyRow(prefixKeys, row, packedKeys, row, kvLength);
                    copyRow(prefixValues, row, packedValues, row, kvLength);
                }
                for (int row = 0; row < batchSize; row++) {
                    copyRow(currentKeys, row, packedKeys, startPosition + row, kvLength);
                    copyRow(currentValues, row, packedValues, startPosition + row, kvLength);
                }

                for (int head = 0; head < numberOfHeads; head++) {
                    int queryOffset = head * config.headSize;
                    int kvOffset = (head / headGroupSize) * config.headSize;
                    lighter.dotProductRows(scores, query, packedKeys, queryOffset, kvOffset, config.headSize,
                            0, capacity, 0);
                    for (int row = 0; row < batchSize; row++) {
                        int visibleLength = startPosition + row + 1;
                        try (TensorRef scoreRow = scores.slice(row); TensorRef outputRow = attended.slice(row)) {
                            compositeOps.scaledSoftMax(new io.teknek.deliverance.tensor2.ScaledSoftMax(attentionScale)
                                    .target(scoreRow)
                                    .offsetAndLength(0, visibleLength));
                            lighter.saxpy(scoreRow, packedValues, outputRow, kvOffset, queryOffset,
                                    config.headSize, 0, 0, visibleLength);
                        }
                    }
                }
                return attended;
            } finally {
                prefixKeys.close();
                prefixValues.close();
            }
        } catch (RuntimeException | Error e) {
            attended.close();
            throw e;
        } finally {
            packedKeys.close();
            packedValues.close();
            scores.close();
        }
    }

    protected final TensorRef[] projectQkv(TensorRef input, ForwardPhase phase) {
        Preconditions.checkArgument(input.dims() == 2 && input.shape().last() == config.embeddingLength,
                "Attention input must be [batch, embeddingLength]");
        int batchSize = input.shape().first();
        TensorRef query = model.makeDenseTensorRef(batchSize, attentionLength);
        TensorRef key = model.makeDenseTensorRef(batchSize, kvLength);
        TensorRef value = model.makeDenseTensorRef(batchSize, kvLength);
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "kvcacheselfattention2.qkv_projection").time()) {
            lighter.dotProductRows(query, input, queryAttentionWeights, 0, config.embeddingLength, 0,
                    attentionLength, 0);
            lighter.dotProductRows(key, input, keyAttentionWeights, 0, config.embeddingLength, 0,
                    kvLength, 0);
            lighter.dotProductRows(value, input, valueAttentionWeights, 0, config.embeddingLength, 0,
                    kvLength, 0);
            return new TensorRef[] {query, key, value};
        } catch (RuntimeException | Error e) {
            query.close();
            key.close();
            value.close();
            throw e;
        }
    }

    protected final TensorRef projectOutput(TensorRef attended, ForwardPhase phase) {
        Preconditions.checkArgument(attended.dims() == 2 && attended.shape().last() == attentionLength,
                "Attention output must be [batch, attentionLength]");
        TensorRef output = model.makeDenseTensorRef(attended.shape().first(), config.embeddingLength);
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "kvcacheselfattention2.output_projection").time()) {
            lighter.dotProductRows(output, attended, outputProjectionWeights, 0, attentionLength, 0,
                    config.embeddingLength, 0);
            return output;
        } catch (RuntimeException | Error e) {
            output.close();
            throw e;
        }
    }

    protected boolean writesCache(CacheExecutionMode mode) {
        return mode == CacheExecutionMode.PREFILL_UPDATE_CACHE
                || mode == CacheExecutionMode.DECODE_UPDATE_CACHE
                || mode == CacheExecutionMode.VERIFY_AND_UPDATE_CACHE;
    }

    private void writeKvRows(KvCacheSession kvSession, CacheExecutionMode mode, TensorRef keys, TensorRef values,
            int startPosition) {
        try (KvWriteCursor writer = kvSession.writer(mode)) {
            for (int row = 0; row < keys.shape().first(); row++) {
                try (TensorRef key = keys.slice(row); TensorRef value = values.slice(row)) {
                    writer.write(layerIndex, startPosition + row, key, value);
                }
            }
        }
    }
}
