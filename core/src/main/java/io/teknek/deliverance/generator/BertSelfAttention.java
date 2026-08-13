package io.teknek.deliverance.generator;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.common.base.Preconditions;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** HF BERT encoder self-attention: full bidirectional attention within each sequence, with key masking. */
public class BertSelfAttention implements SelfAttention {
    private final AbstractModel model;
    private final Config config;
    private final int layerIndex;
    private final Optional<AbstractTensor> queryBias;
    private final Optional<AbstractTensor> keyBias;
    private final Optional<AbstractTensor> valueBias;
    private final AbstractTensor queryWeight;
    private final AbstractTensor keyWeight;
    private final AbstractTensor valueWeight;
    private final Optional<AbstractTensor> outputBias;
    private final AbstractTensor outputWeight;
    private final ConfigurableTensorProvider configurableTensorProvider;
    private final MetricRegistry metricRegistry;
    private final float attentionScale;

    public BertSelfAttention(AbstractModel model, int layerIndex, Optional<AbstractTensor> queryBias,
            Optional<AbstractTensor> keyBias, Optional<AbstractTensor> valueBias, AbstractTensor queryWeight,
            AbstractTensor keyWeight, AbstractTensor valueWeight, Optional<AbstractTensor> outputBias,
            AbstractTensor outputWeight, ConfigurableTensorProvider configurableTensorProvider,
            MetricRegistry metricRegistry) {
        this.model = model;
        this.config = model.getConfig();
        this.layerIndex = layerIndex;
        this.queryBias = queryBias;
        this.keyBias = keyBias;
        this.valueBias = valueBias;
        this.queryWeight = queryWeight;
        this.keyWeight = keyWeight;
        this.valueWeight = valueWeight;
        this.outputBias = outputBias;
        this.outputWeight = outputWeight;
        this.configurableTensorProvider = configurableTensorProvider;
        this.metricRegistry = metricRegistry;
        this.attentionScale = config.attentionMultiplier != null
                ? config.attentionMultiplier
                : (float) (1.0 / StrictMath.sqrt(config.headSize));
        configurableTensorProvider.get().registerModelTensor(queryWeight);
        configurableTensorProvider.get().registerModelTensor(keyWeight);
        configurableTensorProvider.get().registerModelTensor(valueWeight);
        configurableTensorProvider.get().registerModelTensor(outputWeight);
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, int startPosition, KvBufferCache.KvBuffer kvMem,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        return forward(input, 1, (int) input.shape().first(), null, tensorReducer, ForwardPhase.PREFILL);
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, int startPosition, KvBufferCache.KvBuffer kvMem,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer, ForwardPhase phase) {
        return forward(input, 1, (int) input.shape().first(), null, tensorReducer, phase);
    }

    public AbstractTensor forward(AbstractTensor input, int batchSize, int sequenceLength, int[] attentionMask,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer, ForwardPhase phase) {
        Timer timer = InferenceProfiler.timer(metricRegistry, "bertselfattention.forward");
        try (Timer.Context ignored = timer.time()) {
            Preconditions.checkArgument(input.dims() == 2 && input.shape().last() == config.embeddingLength);
            Preconditions.checkArgument(input.shape().first() == (long) batchSize * sequenceLength);
            if (attentionMask != null) {
                Preconditions.checkArgument(attentionMask.length == batchSize * sequenceLength);
            }
            int rows = batchSize * sequenceLength;
            try (AbstractTensor query = model.makeDenseTensor(rows, config.attentionLength);
                 AbstractTensor key = model.makeDenseTensor(rows, config.attentionLength);
                 AbstractTensor value = model.makeDenseTensor(rows, config.attentionLength);
                 AbstractTensor attentionValue = model.makeDenseTensor(rows, config.attentionLength)) {
                projectQkv(input, query, key, value);
                model.emitLayerDebug(layerIndex, "bert_query_projection", query);
                model.emitLayerDebug(layerIndex, "bert_key_projection", key);
                model.emitLayerDebug(layerIndex, "bert_value_projection", value);
                computeBidirectionalAttention(query, key, value, attentionValue, batchSize, sequenceLength,
                        attentionMask);
                model.emitLayerDebug(layerIndex, "bert_attention_value", attentionValue);

                AbstractTensor result = model.makeDenseTensor(rows, config.embeddingLength);
                try (AbstractTensor qAttentionValue = model.maybeQuantizeReadOnly(attentionValue,
                        "bertselfattention.maybe_quantize.output_projection")) {
                    VectorMath.pchunk(0, config.embeddingLength, (chunkStart, chunkSize) ->
                            configurableTensorProvider.get().dotProductChunk(result, qAttentionValue, outputWeight,
                                    0, config.attentionLength, chunkStart, chunkSize),
                            configurableTensorProvider.get().parallelSplitSize(), model.getPool());
                }
                outputBias.ifPresent(bias -> configurableTensorProvider.get().accumulate(result, bias, 0,
                        config.embeddingLength));
                tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(result)));
                model.emitLayerDebug(layerIndex, "bert_attention_output", result);
                return result;
            }
        }
    }

    private void projectQkv(AbstractTensor input, AbstractTensor query, AbstractTensor key, AbstractTensor value) {
        int splitSize = configurableTensorProvider.get().parallelSplitSize();
        AbstractTensor[] outputs = { query, key, value };
        AbstractTensor[] weights = { queryWeight, keyWeight, valueWeight };
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "bertselfattention.qkv_projection").time()) {
            VectorMath.pchunk(0, config.attentionLength, (chunkStart, chunkLength) ->
                    configurableTensorProvider.get().dotProductBatchChunk(outputs, input, weights, 0,
                            config.embeddingLength, chunkStart, chunkLength), splitSize, model.getPool());
        }
        queryBias.ifPresent(bias -> configurableTensorProvider.get().accumulate(query, bias, 0, config.attentionLength));
        keyBias.ifPresent(bias -> configurableTensorProvider.get().accumulate(key, bias, 0, config.attentionLength));
        valueBias.ifPresent(bias -> configurableTensorProvider.get().accumulate(value, bias, 0, config.attentionLength));
    }

    private void computeBidirectionalAttention(AbstractTensor query, AbstractTensor key, AbstractTensor value,
            AbstractTensor output, int batchSize, int sequenceLength, int[] attentionMask) {
        VectorMath.pfor(0, batchSize * sequenceLength, queryRow -> {
            int batch = queryRow / sequenceLength;
            for (int head = 0; head < config.numberOfHeads; head++) {
                int headOffset = head * config.headSize;
                float maxScore = Float.NEGATIVE_INFINITY;
                float[] scores = new float[sequenceLength];
                for (int keyToken = 0; keyToken < sequenceLength; keyToken++) {
                    int keyRow = batch * sequenceLength + keyToken;
                    if (attentionMask != null && attentionMask[keyRow] == 0) {
                        scores[keyToken] = Float.NEGATIVE_INFINITY;
                        continue;
                    }
                    float score = 0.0f;
                    for (int i = 0; i < config.headSize; i++) {
                        score += query.get(queryRow, headOffset + i) * key.get(keyRow, headOffset + i);
                    }
                    score *= attentionScale;
                    scores[keyToken] = score;
                    maxScore = Math.max(maxScore, score);
                }
                float sum = 0.0f;
                for (int keyToken = 0; keyToken < sequenceLength; keyToken++) {
                    if (scores[keyToken] == Float.NEGATIVE_INFINITY) {
                        continue;
                    }
                    float exp = (float) Math.exp(scores[keyToken] - maxScore);
                    scores[keyToken] = exp;
                    sum += exp;
                }
                for (int i = 0; i < config.headSize; i++) {
                    float weighted = 0.0f;
                    if (sum != 0.0f) {
                        for (int keyToken = 0; keyToken < sequenceLength; keyToken++) {
                            if (scores[keyToken] == Float.NEGATIVE_INFINITY) {
                                continue;
                            }
                            int valueRow = batch * sequenceLength + keyToken;
                            weighted += (scores[keyToken] / sum) * value.get(valueRow, headOffset + i);
                        }
                    }
                    output.set(weighted, queryRow, headOffset + i);
                }
            }
        }, model.getPool());
    }
}
