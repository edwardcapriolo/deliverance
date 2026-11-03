package io.teknek.reckoning.model.bert;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.embedding.PoolingType;
import io.teknek.reckoning.grace.EncodeOptions;
import io.teknek.reckoning.grace.Encoding;
import io.teknek.reckoning.grace.PreTrainedTokenizer;
import io.teknek.reckoning.generator.ForwardPhase;
import io.teknek.reckoning.math.WrappedForkJoinPool;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.tensorparallel.TensorParallelCollectives;
import io.teknek.reckoning.model.tensorparallel.TensorParallelContext;
import io.teknek.reckoning.safetensors.Config;
import io.teknek.reckoning.safetensors.WeightLoader;
import io.teknek.reckoning.tensor.KvBufferCacheSettings;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import io.teknek.reckoning.tensorlib.PlannedTensorRef;
import io.teknek.reckoning.tensor.kv.KvCacheSession;
import io.teknek.reckoning.toolcallparser.ToolCallParser;

import java.util.Optional;
import java.util.SortedMap;
import java.util.TreeMap;

/** Thin RoBERTa specialization over the shared BERT-family encoder. */
public class RobertaModel extends BertFamilyModel {
    private RobertaClassificationHead classificationHead;

    public RobertaModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer tokenizer,
            DType workingDType, DType workingQType, Optional<DType> modelQType,
            MetricRegistry metricRegistry, KvBufferCacheSettings kvBufferCacheSettings,
            ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization, Lighter lighter) {
        super(inferenceType, c, w, tokenizer, workingDType, workingQType, modelQType,
                metricRegistry, kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext,
                tensorParallelCollectives, outputHeadQuantization, lighter);
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

    private void loadClassificationHead() {
        TensorRef denseWeight = registerModelTensorRef(loadClassifierRef("classifier.dense.weight"));
        TensorRef denseBias = registerModelTensorRef(loadClassifierRef("classifier.dense.bias"));
        TensorRef outputWeight = registerModelTensorRef(loadClassifierRef("classifier.out_proj.weight"));
        TensorRef outputBias = registerModelTensorRef(loadClassifierRef("classifier.out_proj.bias"));
        classificationHead = new RobertaClassificationHead(denseWeight, denseBias, outputWeight, outputBias,
                config.embeddingLength, outputWeight.shape().first());
    }

    protected int positionPaddingId() {
        return config instanceof BertConfig bertConfig ? bertConfig.padTokenId : 1;
    }

    private TensorRef loadClassifierRef(String name) {
        return weights.isWeightPresent(name) ? weights.loadRef(name) : loadWeightRef(name);
    }


    @Override
    protected SortedMap<String, Float> classifyInput(String input, PoolingType poolingType) {
        if (classificationHead == null) {
            loadClassificationHead();
        }
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
