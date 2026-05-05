package io.teknek.deliverance.generator;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.gemma4.Gemma4Config;
import io.teknek.deliverance.model.gemma4.Gemma4Model;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import net.jafama.FastMath;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

public class Gemma4CausalSelfAttention extends CausalSelfAttention {
    private final AbstractModel model;
    private final Gemma4Config config;
    private final int layerIndex;
    private final String layerType;
    private final boolean slidingAttention;
    private final boolean kvSharedLayer;
    private final boolean storeSharedKv;
    private final int sharedKvSourceLayer;
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

    public Gemma4CausalSelfAttention(
            AbstractModel model,
            int layerIndex,
            String layerType,
            boolean kvSharedLayer,
            boolean storeSharedKv,
            int sharedKvSourceLayer,
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
        this.kvSharedLayer = kvSharedLayer;
        this.storeSharedKv = storeSharedKv;
        this.sharedKvSourceLayer = sharedKvSourceLayer;
        this.headDim = config.getLayerHeadDim(layerType);
        this.rotaryDim = config.rotaryDimensionsByLayerType.get(layerType);
        this.numberOfHeads = config.numberOfHeads;
        this.numberOfKeyValueHeads = config.getLayerKeyValueHeads(layerType);
        this.numberOfKeyValueGroups = numberOfHeads / numberOfKeyValueHeads;
        this.queryLength = numberOfHeads * headDim;
        this.kvLength = numberOfKeyValueHeads * headDim;
        this.slidingWindow = config.slidingWindow == null ? config.contextLength : config.slidingWindow;
        this.ropeFreqs = config.ropeFreqsByLayerType.get(layerType);
        this.queryWeights = queryWeights;
        this.queryNormWeights = queryNormWeights;
        this.keyWeights = keyWeights;
        this.valueWeights = valueWeights;
        this.keyNormWeights = keyNormWeights;
        this.outputProjectionWeights = outputProjectionWeights;
        this.configurableTensorProvider = configurableTensorProvider;
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, int startPosition, KvBufferCache.KvBuffer kvMem,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        int batchSize = input.shape().first();
        try (
                AbstractTensor queryBatch = model.makeDenseTensor(batchSize, queryLength);
                AbstractTensor keyBatch = model.makeDenseTensor(batchSize, kvLength);
                AbstractTensor valueBatch = model.makeDenseTensor(batchSize, kvLength);
                AbstractTensor valueOutput = model.makeDenseTensor(batchSize, queryLength)
        ) {
            project(input, queryBatch, queryWeights, queryLength);
            Gemma4RmsNormSupport.applyInPlace(queryBatch, numberOfHeads, headDim, config.layerNormEps, queryNormWeights);

            if (kvSharedLayer) {
                Gemma4Model.SharedKeyValues shared = ((Gemma4Model) model).getSharedKeyValues(sharedKvSourceLayer);
                keyBatch.copyFrom(shared.key(), 0, 0, (int) shared.key().size());
                valueBatch.copyFrom(shared.value(), 0, 0, (int) shared.value().size());
            } else {
                project(input, keyBatch, keyWeights.orElseThrow(), kvLength);
                project(input, valueBatch, valueWeights.orElse(keyWeights.orElseThrow()), kvLength);
                Gemma4RmsNormSupport.applyInPlace(keyBatch, numberOfKeyValueHeads, headDim, config.layerNormEps,
                        keyNormWeights.orElseThrow());
                Gemma4RmsNormSupport.applyInPlace(valueBatch, numberOfKeyValueHeads, headDim, config.layerNormEps, null);
            }

            applyRope(queryBatch, numberOfHeads, startPosition);
            if (!kvSharedLayer) {
                applyRope(keyBatch, numberOfKeyValueHeads, startPosition);
                if (storeSharedKv) {
                    ((Gemma4Model) model).putSharedKeyValues(layerIndex, keyBatch, valueBatch);
                }
            }

            for (int position = startPosition, batchIndex = 0; position < startPosition + batchSize; position++, batchIndex++) {
                AbstractTensor keyTensor = kvMem.getKeyTensorForPosition(layerIndex, position);
                AbstractTensor valueTensor = kvMem.getValTensorForPosition(layerIndex, position);
                copyKvRow(keyBatch, valueBatch, batchIndex, keyTensor, valueTensor);

                AbstractTensor[] keyPages = kvMem.getKeyTensorsUptoPosition(layerIndex, position);
                AbstractTensor[] valuePages = kvMem.getValTensorsUptoPosition(layerIndex, position);
                int windowStart = slidingAttention ? Math.max(0, position - slidingWindow + 1) : 0;

                for (int head = 0; head < numberOfHeads; head++) {
                    int kvHead = head / numberOfKeyValueGroups;
                    int queryOffset = head * headDim;
                    int kvOffset = kvHead * headDim;

                    float[] scores = new float[position - windowStart + 1];
                    int scoreIndex = 0;
                    int globalOffset = 0;
                    for (AbstractTensor keyPage : keyPages) {
                        int limit = Math.min(keyPage.shape().first(), (position + 1) - globalOffset);
                        for (int row = 0; row < limit; row++) {
                            int absolutePosition = globalOffset + row;
                            if (absolutePosition >= windowStart) {
                                scores[scoreIndex++] = score(queryBatch, batchIndex, queryOffset, keyPage, row, kvOffset);
                            }
                        }
                        globalOffset += keyPage.shape().first();
                    }

                    softmax(scores);

                    scoreIndex = 0;
                    globalOffset = 0;
                    for (AbstractTensor valuePage : valuePages) {
                        int limit = Math.min(valuePage.shape().first(), (position + 1) - globalOffset);
                        for (int row = 0; row < limit; row++) {
                            int absolutePosition = globalOffset + row;
                            if (absolutePosition >= windowStart) {
                                float weight = scores[scoreIndex++];
                                for (int i = 0; i < headDim; i++) {
                                    float accum = valueOutput.get(batchIndex, queryOffset + i);
                                    accum += weight * valuePage.get(row, kvOffset + i);
                                    valueOutput.set(accum, batchIndex, queryOffset + i);
                                }
                            }
                        }
                        globalOffset += valuePage.shape().first();
                    }
                }

                keyTensor.close();
                valueTensor.close();
                closeAll(keyPages);
                closeAll(valuePages);
            }

            AbstractTensor result = model.makeDenseTensor(batchSize, config.embeddingLength);
            try (AbstractTensor valueQ = model.maybeQuantize(valueOutput)) {
                configurableTensorProvider.get().dotProductChunk(result, valueQ, outputProjectionWeights, 0, queryLength, 0,
                        config.embeddingLength);
                tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(result)));
            }
            return result;
        }
    }

    private void project(AbstractTensor input, AbstractTensor output, AbstractTensor weights, int outputLength) {
        configurableTensorProvider.get().dotProductChunk(output, input, weights, 0, config.embeddingLength, 0, outputLength);
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

    private float score(AbstractTensor queryBatch, int batchIndex, int queryOffset, AbstractTensor keyPage, int row, int kvOffset) {
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

    private void softmax(float[] scores) {
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
}
