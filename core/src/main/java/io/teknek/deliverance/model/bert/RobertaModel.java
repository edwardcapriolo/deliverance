package io.teknek.deliverance.model.bert;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.classifier.ClassifyOutput;
import io.teknek.deliverance.embedding.PoolingType;
import io.teknek.deliverance.grace.EncodeOptions;
import io.teknek.deliverance.grace.Encoding;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensorlib.PlannedTensorRef;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.util.Optional;
import java.util.SortedMap;
import java.util.TreeMap;

/** Thin RoBERTa specialization over the shared BERT-family encoder. */
public class RobertaModel extends BertFamilyModel {
    private RobertaClassificationHead classificationHead;

    public RobertaModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer tokenizer,
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
    protected String[] weightPrefixes() {
        return new String[] { "roberta." };
    }

    @Override
    protected BertInput inputForEncoding(Encoding encoding) {
        int paddingId = positionPaddingId();
        return new BertInput(encoding.inputIds(), encoding.attentionMask(), null, null, 1,
                encoding.length(), (inputIds, attentionMask, batchSize, sequenceLength) -> {
                    int[] positions = new int[inputIds.length];
                    int position = paddingId;
                    for (int index = 0; index < inputIds.length; index++) {
                        if (inputIds[index] == paddingId) {
                            positions[index] = paddingId;
                        } else {
                            positions[index] = ++position;
                        }
                    }
                    return positions;
                });
    }

    @Override
    protected ClassifyOutput loadClassifierWeights() {
        if (!config.isClassifier()) {
            throw new UnsupportedOperationException("Classification not supported by this model");
        }
        TensorRef denseWeight = registerModelTensorRef(loadClassifierRef("classifier.dense.weight"));
        TensorRef denseBias = registerModelTensorRef(loadClassifierRef("classifier.dense.bias"));
        TensorRef outputWeight = registerModelTensorRef(loadClassifierRef("classifier.out_proj.weight"));
        TensorRef outputBias = registerModelTensorRef(loadClassifierRef("classifier.out_proj.bias"));
        classificationHead = new RobertaClassificationHead(denseWeight, denseBias, outputWeight, outputBias,
                config.embeddingLength, outputWeight.shape().first());
        return new ClassifyOutput() {
            @Override
            public AbstractTensor getClassificationWeights() {
                throw new UnsupportedOperationException("RoBERTa uses a multi-stage classification head");
            }

            @Override
            public Optional<AbstractTensor> getClassificationBias() {
                return Optional.empty();
            }
        };
    }

    protected int positionPaddingId() {
        return config instanceof BertConfig bertConfig ? bertConfig.padTokenId : 1;
    }

    private TensorRef loadClassifierRef(String name) {
        return weights.isWeightPresent(name) ? weights.loadRef(name) : loadWeightRef(name);
    }

    @Override
    protected io.teknek.deliverance.generator.SampleOutput loadOutputWeights() {
        throw new UnsupportedOperationException();
    }

    @Override
    protected SortedMap<String, Float> classifyInput(String input, PoolingType poolingType) {
        EncodeOptions options = EncodeOptions.defaults();
        Encoding encoding = tokenizer.encode(input, options);
        BertInput modelInput = inputForEncoding(encoding);
        try (KvCacheSession session = newKvCacheSession()) {
            TensorRef embeddings = bertEmbeddingsRef(modelInput);
            PlannedTensorRef sequence = forward(plannedInputEmbeddingsRef("roberta_input_embeddings", embeddings,
                    ForwardPhase.PREFILL), 0, session, Optional.empty(), ForwardPhase.PREFILL,
                    modelInput.batchSize(), modelInput.sequenceLength(), modelInput.attentionMask());
            try (TensorRef logits = classificationHead.forward(sequence.tensor(), lighter, this::makeDenseTensorRef)) {
                return probabilities(logits);
            }
        }
    }

    private SortedMap<String, Float> probabilities(TensorRef logits) {
        int labels = (int) logits.shape().last();
        float max = Float.NEGATIVE_INFINITY;
        for (int index = 0; index < labels; index++) {
            max = Math.max(max, logits.get(0, index));
        }
        float total = 0.0f;
        float[] values = new float[labels];
        for (int index = 0; index < labels; index++) {
            values[index] = (float) Math.exp(logits.get(0, index) - max);
            total += values[index];
        }
        SortedMap<String, Float> result = new TreeMap<>();
        for (int index = 0; index < labels; index++) {
            result.put(config.classifcationLabels.get().inverse().get(index), values[index] / total);
        }
        return result;
    }
}
