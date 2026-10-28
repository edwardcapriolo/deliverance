package io.teknek.deliverance.model.bert;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.classifier.ClassifyOutput;
import io.teknek.deliverance.embedding.PoolingLayer;
import io.teknek.deliverance.embedding.PoolingType;
import io.teknek.deliverance.embedding.SentenceTransformersPooling;
import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.generator.SampleOutput;
import io.teknek.deliverance.grace.EncodeOptions;
import io.teknek.deliverance.grace.Encoding;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.grace.TruncationOptions;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor2.TensorRefBackedTensor;
import io.teknek.deliverance.tensorlib.PlannedTensorRef;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.util.Optional;

/** BERT-specific pooler and sequence-classification behavior over the shared family encoder. */
public class BertModel extends BertFamilyModel {
    private TensorRef poolingWeightRef;
    private TensorRef poolingBiasRef;

    public BertModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer tokenizer,
            DType workingDType, DType workingQType, Optional<DType> modelQType,
            ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
            TensorAllocator tensorAllocator, KvBufferCacheSettings kvBufferCacheSettings,
            ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization) {
        super(inferenceType, c, w, tokenizer, workingDType, workingQType, modelQType,
                configurableTensorProvider, metricRegistry, tensorAllocator, kvBufferCacheSettings, toolCallParser,
                pool, tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
    }

    @Override
    protected float[] timedEmbedding(String input, PoolingType poolingType) {
        EncodeOptions encodeOptions = embeddingRecipe.maxSequenceLength().stream()
                .mapToObj(maxLength -> EncodeOptions.defaults().withTruncation(TruncationOptions.maxLength(
                        Math.min(maxLength, config.contextLength))))
                .findFirst()
                .orElseGet(EncodeOptions::defaults);
        Encoding encoding = tokenizer.encode(input, encodeOptions);
        if (encoding.length() >= config.contextLength) {
            throw new IllegalArgumentException("Encoded input length " + encoding.length()
                    + " exceeds context length " + config.contextLength);
        }
        BertInput bertInput = inputForEncoding(encoding);
        try (KvCacheSession kvSession = newKvCacheSession()) {
            TensorRef inputEmbeddings = bertEmbeddingsRef(bertInput);
            PlannedTensorRef planned = forward(plannedInputEmbeddingsRef("bert_input_embeddings", inputEmbeddings,
                    ForwardPhase.PREFILL), 0, kvSession, Optional.empty(), ForwardPhase.PREFILL,
                    bertInput.batchSize(), bertInput.sequenceLength(), bertInput.attentionMask());
            try (TensorRefBackedTensor tokenEmbeddings = new TensorRefBackedTensor(planned.tensor())) {
                if (poolingType == PoolingType.MODEL) {
                    if (poolingWeightRef == null || poolingBiasRef == null) {
                        throw new UnsupportedOperationException("no pooling layer for this model");
                    }
                    try (TensorRef last = planned.tensor().slice(bertInput.sequenceLength() - 1);
                         TensorRef pooled = makeDenseTensorRef(1, config.embeddingLength)) {
                        lighter.dotProductRows(pooled, last, poolingWeightRef, 0, config.embeddingLength,
                                0, config.embeddingLength, 0);
                        for (int column = 0; column < config.embeddingLength; column++) {
                            pooled.set(ActivationFunction.eval(config.activationFunction,
                                    pooled.get(0, column) + poolingBiasRef.get(0, column)), 0, column);
                        }
                        float[] output = new float[config.embeddingLength];
                        for (int column = 0; column < config.embeddingLength; column++) {
                            output[column] = pooled.get(0, column);
                        }
                        return output;
                    }
                }
                SentenceTransformersPooling.Mode[] modes = embeddingRecipe.poolingModes()
                        .toArray(SentenceTransformersPooling.Mode[]::new);
                float[] embedding = SentenceTransformersPooling.pool(tokenEmbeddings, bertInput.attentionMask(),
                        bertInput.batchSize(), bertInput.sequenceLength(), modes)[0];
                if (embeddingRecipe.normalize()) {
                    SentenceTransformersPooling.normalize(embedding);
                }
                return embedding;
            }
        }
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        throw new UnsupportedOperationException();
    }

    @Override
    protected ClassifyOutput loadClassifierWeights() {
        if (config.isClassifier()) {
            final AbstractTensor classifierWeight = loadWeight("classifier.weight");
            final AbstractTensor classifierBias = loadWeight("classifier.bias");
            return new ClassifyOutput() {
                @Override
                public AbstractTensor getClassificationWeights() {
                    return classifierWeight;
                }

                @Override
                public Optional<AbstractTensor> getClassificationBias() {
                    return Optional.of(classifierBias);
                }
            };
        }
        throw new UnsupportedOperationException("Classification not supported by this model");
    }

    @Override
    protected PoolingLayer loadPoolingWeights() {
        if (!weights.isWeightPresent("pooler.dense.weight") && !weights.isWeightPresent("bert.pooler.dense.weight")) {
            return null;
        }
        final AbstractTensor poolerDenseWeight = loadWeight("pooler.dense.weight");
        final AbstractTensor poolerDenseBias = loadWeight("pooler.dense.bias");
        poolingWeightRef = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                "pooler.dense.weight", this::loadWeightRef, modelQType.orElse(modelDType)));
        poolingBiasRef = registerModelTensorRef(loadWeightRef("pooler.dense.bias"));
        return new PoolingLayer() {
            public AbstractTensor getPoolingWeights() {
                return poolerDenseWeight;
            }

            public Optional<AbstractTensor> getPoolingBias() {
                return Optional.of(poolerDenseBias);
            }
        };
    }
}
