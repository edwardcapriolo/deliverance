package io.teknek.deliverance.model.bert;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.embedding.PoolingType;
import io.teknek.deliverance.generator.BertTransformerBlock;
import io.teknek.deliverance.generator.EmbedInput;
import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.generator.LayerNorm;
import io.teknek.deliverance.grace.EncodeOptions;
import io.teknek.deliverance.grace.Encoding;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.tensorparallel.SingleRankTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.safetensors.SafeTensorWriter;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.Mockito;

import java.nio.file.Path;
import java.nio.file.Files;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.SortedMap;
import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;

/** Ports the first shape-only case from Hugging Face transformers tests/models/bert/test_modeling_bert.py. */
public class BertModelingPortedTest {
    private static final int BATCH_SIZE = 13;
    private static final int SEQ_LENGTH = 7;
    private static final int VOCAB_SIZE = 99;
    private static final int HIDDEN_SIZE = 32;
    private static final int NUM_LAYERS = 2;
    private static final int NUM_HEADS = 4;
    private static final int INTERMEDIATE_SIZE = 37;
    private static final int TYPE_VOCAB_SIZE = 16;
    private static final int MAX_POSITION_EMBEDDINGS = 512;

    @TempDir
    Path tempDir;

    @Test
    public void createAndCheckModel() throws Exception {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("bert-tiny"));
        try (BertModel model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor hidden = model.batchForward(inputIds(), 0, kv)) {

            // HF returns [batch, seq, hidden]; Deliverance flattens sequence rows to [batch * seq, hidden].
            assertEquals(BATCH_SIZE * SEQ_LENGTH, hidden.shape().first());
            assertEquals(HIDDEN_SIZE, hidden.shape().last());
        }
    }

    @Test
    public void createAndCheckForSequenceClassification() throws Exception {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("bert-tiny-classifier"));
        try (BertModel model = loadTinyModel(modelDir, AbstractModel.InferenceType.FULL_CLASSIFICATION)) {
            SortedMap<String, Float> scores = model.classify("tokenizer is mocked so text is ignored", PoolingType.MODEL);

            assertEquals(3, scores.size());
        }
    }

    @Test
    public void bertInputDefaultsMatchHfBroadcasts() {
        BertInput input = new BertInput(new int[] { 4, 5, 6, 7, 8, 9 }, null, null, null, 2, 3);

        assertArrayEquals(new int[] { 1, 1, 1, 1, 1, 1 }, input.attentionMask());
        assertArrayEquals(new int[] { 0, 0, 0, 0, 0, 0 }, input.tokenTypeIds());
        assertArrayEquals(new int[] { 0, 1, 2, 0, 1, 2 }, input.positionIds());
    }

    @Test
    public void bertEmbeddingsUseExplicitTokenTypeAndPositionIds() throws Exception {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("bert-tiny-embeddings"));
        int[] inputIds = { 3, 11, 19 };
        int[] tokenTypeIds = { 1, 2, 3 };
        int[] positionIds = { 7, 6, 5 };

        try (BertModel model = loadTinyModel(modelDir);
             AbstractTensor output = model.bertEmbeddings(BertInput.singleSequence(inputIds, null, tokenTypeIds, positionIds))) {

            assertEquals(inputIds.length, output.shape().first());
            assertEquals(HIDDEN_SIZE, output.shape().last());
            for (int row = 0; row < inputIds.length; row++) {
                float[] expected = expectedEmbeddingRow(inputIds[row], tokenTypeIds[row], positionIds[row]);
                for (int col = 0; col < HIDDEN_SIZE; col++) {
                    assertEquals(expected[col], output.get(row, col), 1.0e-5f, "row=" + row + " col=" + col);
                }
            }
        }
    }

    @Test
    public void printsBertEmbeddingTensorPlan() throws Exception {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("bert-tiny-embedding-plan"));
        BertInput input = BertInput.singleSequence(new int[] { 3, 11, 19 }, new int[] { 1, 1, 1 },
                new int[] { 0, 1, 0 }, new int[] { 0, 2, 4 });
        try (BertModel model = loadTinyModel(modelDir)) {
            String plan = model.bertEmbeddingPathPlan(input);
            Path capture = Path.of("target", "bert-embedding-tensor-plan.txt");
            Files.createDirectories(capture.getParent());
            Files.writeString(capture, plan);
            System.out.println("BERT embedding TensorPlan written to " + capture.toAbsolutePath());
            System.out.println(plan);

            assertEquals(true, plan.contains("bert_embeddings.gather_add"));
            assertEquals(true, plan.contains("bert_embeddings.layernorm"));
            assertEquals(true, plan.contains("model.weights.embeddings.word_embeddings.weight"));
            assertEquals(true, plan.contains("model.weights.embeddings.token_type_embeddings.weight"));
            assertEquals(true, plan.contains("model.weights.embeddings.position_embeddings.weight"));
            assertEquals(true, plan.contains("model.weights.embeddings.LayerNorm.weight"));
            assertEquals(true, plan.contains("model.weights.embeddings.LayerNorm.bias"));
            assertEquals(true, plan.contains("word_embeddings"));
            assertEquals(true, plan.contains("token_type_embeddings"));
            assertEquals(true, plan.contains("position_embeddings"));
            assertEquals(true, plan.contains("embedding"));
        }
    }

    @Test
    public void bertTransformerBlockAppliesResidualBeforeLayerNorm() throws Exception {
        Path modelDir = writeTinyCheckpoint(tempDir.resolve("bert-tiny-block-order"));
        try (BertModel model = loadTinyModel(modelDir);
             KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor input = tensorFromFormula(model, 2, HIDDEN_SIZE, 0.25f)) {
            LayerNorm postAttentionNorm = new LayerNorm(model, zeros(1, HIDDEN_SIZE), ones(1, HIDDEN_SIZE), new MetricRegistry());
            LayerNorm postFfNorm = new LayerNorm(model, zeros(1, HIDDEN_SIZE), ones(1, HIDDEN_SIZE), new MetricRegistry());
            BertTransformerBlock block = new BertTransformerBlock(model, 0,
                    (hidden, startPosition, kvMem, reducer) -> tensorFromFormula(model, 2, HIDDEN_SIZE, 1.25f),
                    postAttentionNorm,
                    (hidden, reducer) -> tensorFromFormula(model, 2, HIDDEN_SIZE, -0.75f),
                    postFfNorm,
                    new ConfigurableTensorProvider(new NaiveTensorOperations()));

            float[][] attentionResidual = add(input, formulaRows(2, HIDDEN_SIZE, 1.25f));
            float[][] attentionOutput = layerNorm(attentionResidual);
            float[][] ffResidual = add(attentionOutput, formulaRows(2, HIDDEN_SIZE, -0.75f));
            float[][] expected = layerNorm(ffResidual);

            try (AbstractTensor output = block.forward(input, 0, kv, Optional.empty(), ForwardPhase.PREFILL)) {
                for (int row = 0; row < 2; row++) {
                    for (int col = 0; col < HIDDEN_SIZE; col++) {
                        assertEquals(expected[row][col], output.get(row, col), 1.0e-5f,
                                "row=" + row + " col=" + col);
                    }
                }
            }
        }
    }

    @Test
    @Disabled("Deliverance BertModel does not support decoder/cross-attention mode.")
    public void createAndCheckModelAsDecoder() {
    }

    @Test
    @Disabled("Deliverance does not implement BertLMHeadModel for causal LM.")
    public void createAndCheckForCausalLm() {
    }

    @Test
    @Disabled("Deliverance does not implement BertForMaskedLM/prediction logits.")
    public void createAndCheckForMaskedLm() {
    }

    @Test
    @Disabled("Deliverance does not implement decoder BertLMHeadModel with cross-attention.")
    public void createAndCheckModelForCausalLmAsDecoder() {
    }

    @Test
    @Disabled("Deliverance BERT path does not expose decoder past-key-values behavior.")
    public void createAndCheckDecoderModelPastLargeInputs() {
    }

    @Test
    @Disabled("Deliverance does not implement BertForNextSentencePrediction head.")
    public void createAndCheckForNextSequencePrediction() {
    }

    @Test
    @Disabled("Deliverance does not implement BertForPreTraining heads.")
    public void createAndCheckForPretraining() {
    }

    @Test
    @Disabled("Deliverance does not implement BertForQuestionAnswering start/end logits.")
    public void createAndCheckForQuestionAnswering() {
    }

    @Test
    @Disabled("Deliverance does not implement BertForTokenClassification logits.")
    public void createAndCheckForTokenClassification() {
    }

    @Test
    @Disabled("Deliverance does not implement BertForMultipleChoice logits.")
    public void createAndCheckForMultipleChoice() {
    }

    @Test
    @Tag("large-model")
    public void testInferenceNoHeadAbsoluteEmbedding() {
        ModelFetcher fetch = new ModelFetcher("google-bert", "bert-base-uncased");
        java.io.File modelRoot = fetch.maybeDownload();

        int[] inputIds = {0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2};
        int[] attentionMask = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        float[][] expected = {
                {0.4249f, 0.1008f, 0.7531f},
                {0.3771f, 0.1188f, 0.7467f},
                {0.4152f, 0.1098f, 0.7108f}
        };

        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(1));
             BertModel model = new BertModel(AbstractModel.InferenceType.FULL_EMBEDDING,
                     JsonUtils.om.readValue(modelRoot.toPath().resolve("config.json").toFile(), BertConfig.class),
                     new DefaultWeightLoader(modelRoot), Mockito.mock(PreTrainedTokenizer.class),
                     DType.F32, DType.F32, Optional.empty(),
                     new ConfigurableTensorProvider(new NaiveTensorOperations()), metrics, allocator,
                     new KvBufferCacheSettings(true), new DefaultToolCallParser(), pool,
                     new StaticTensorParallelContext(0, 1), new SingleRankTensorParallelCollectives(), Optional.empty());
             KvBufferCache.KvBuffer kv = model.newKvBuffer()) {
            model.init();
            try (AbstractTensor output = model.batchForward(BertInput.singleSequence(inputIds, attentionMask, null, null), kv)) {
                assertEquals(11, output.shape().first());
                assertEquals(768, output.shape().last());
                for (int row = 0; row < 3; row++) {
                    for (int col = 0; col < 3; col++) {
                        assertEquals(expected[row][col], output.get(row + 1, col + 1), 1.0e-4f,
                                "row=" + row + " col=" + col);
                    }
                }
            }
        } catch (java.io.IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static BertModel loadTinyModel(Path modelDir) {
        return loadTinyModel(modelDir, AbstractModel.InferenceType.FULL_EMBEDDING);
    }

    private static BertModel loadTinyModel(Path modelDir, AbstractModel.InferenceType inferenceType) {
        PreTrainedTokenizer tokenizer = Mockito.mock(PreTrainedTokenizer.class);
        when(tokenizer.encode(anyString(), any(EncodeOptions.class))).thenReturn(new Encoding(
                inputIds(), onesArray(BATCH_SIZE * SEQ_LENGTH), zerosArray(BATCH_SIZE * SEQ_LENGTH)));
        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(1));
        BertModel model = new BertModel(inferenceType, config(), new DefaultWeightLoader(modelDir.toFile()), tokenizer,
                DType.F32, DType.F32, Optional.empty(), new ConfigurableTensorProvider(new NaiveTensorOperations()),
                metrics, allocator, new KvBufferCacheSettings(true), new DefaultToolCallParser(), pool,
                new StaticTensorParallelContext(0, 1), new SingleRankTensorParallelCollectives(), Optional.empty());
        model.init();
        return model;
    }

    private static Path writeTinyCheckpoint(Path dir) throws Exception {
        Files.createDirectories(dir);
        Map<String, Object> configJson = new LinkedHashMap<>();
        configJson.put("model_type", "bert");
        configJson.put("vocab_size", VOCAB_SIZE);
        configJson.put("hidden_size", HIDDEN_SIZE);
        configJson.put("num_hidden_layers", NUM_LAYERS);
        configJson.put("num_attention_heads", NUM_HEADS);
        configJson.put("intermediate_size", INTERMEDIATE_SIZE);
        configJson.put("hidden_act", "gelu");
        configJson.put("max_position_embeddings", MAX_POSITION_EMBEDDINGS);
        configJson.put("type_vocab_size", TYPE_VOCAB_SIZE);
        configJson.put("layer_norm_eps", 1.0e-12);
        configJson.put("cls_token", 101);
        configJson.put("sep_token", 102);
        configJson.put("label2id", Map.of("LABEL_0", 0, "LABEL_1", 1, "LABEL_2", 2));
        JsonUtils.om.writeValue(dir.resolve("config.json").toFile(), configJson);

        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put("embeddings.word_embeddings.weight", matrix(VOCAB_SIZE, HIDDEN_SIZE, 1));
        tensors.put("embeddings.token_type_embeddings.weight", matrix(TYPE_VOCAB_SIZE, HIDDEN_SIZE, 2));
        tensors.put("embeddings.position_embeddings.weight", matrix(MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE, 3));
        tensors.put("embeddings.LayerNorm.weight", ones(1, HIDDEN_SIZE));
        tensors.put("embeddings.LayerNorm.bias", zeros(1, HIDDEN_SIZE));
        tensors.put("pooler.dense.weight", matrix(HIDDEN_SIZE, HIDDEN_SIZE, 4));
        tensors.put("pooler.dense.bias", zeros(1, HIDDEN_SIZE));
        tensors.put("classifier.weight", matrix(3, HIDDEN_SIZE, 5));
        tensors.put("classifier.bias", zeros(1, 3));

        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            String prefix = "encoder.layer." + layer + ".";
            tensors.put(prefix + "attention.self.query.weight", matrix(HIDDEN_SIZE, HIDDEN_SIZE, 10 + layer));
            tensors.put(prefix + "attention.self.query.bias", zeros(1, HIDDEN_SIZE));
            tensors.put(prefix + "attention.self.key.weight", matrix(HIDDEN_SIZE, HIDDEN_SIZE, 20 + layer));
            tensors.put(prefix + "attention.self.key.bias", zeros(1, HIDDEN_SIZE));
            tensors.put(prefix + "attention.self.value.weight", matrix(HIDDEN_SIZE, HIDDEN_SIZE, 30 + layer));
            tensors.put(prefix + "attention.self.value.bias", zeros(1, HIDDEN_SIZE));
            tensors.put(prefix + "attention.output.dense.weight", matrix(HIDDEN_SIZE, HIDDEN_SIZE, 40 + layer));
            tensors.put(prefix + "attention.output.dense.bias", zeros(1, HIDDEN_SIZE));
            tensors.put(prefix + "attention.output.LayerNorm.weight", ones(1, HIDDEN_SIZE));
            tensors.put(prefix + "attention.output.LayerNorm.bias", zeros(1, HIDDEN_SIZE));
            tensors.put(prefix + "intermediate.dense.weight", matrix(INTERMEDIATE_SIZE, HIDDEN_SIZE, 50 + layer));
            tensors.put(prefix + "intermediate.dense.bias", zeros(1, INTERMEDIATE_SIZE));
            tensors.put(prefix + "output.dense.weight", matrix(HIDDEN_SIZE, INTERMEDIATE_SIZE, 60 + layer));
            tensors.put(prefix + "output.dense.bias", zeros(1, HIDDEN_SIZE));
            tensors.put(prefix + "output.LayerNorm.weight", ones(1, HIDDEN_SIZE));
            tensors.put(prefix + "output.LayerNorm.bias", zeros(1, HIDDEN_SIZE));
        }

        SafeTensorWriter.writeModel(dir, Map.of("format", "pt"), tensors, 1 << 28);
        return dir;
    }

    private static BertConfig config() {
        return new BertConfig(MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_HEADS, NUM_LAYERS,
                1.0e-12f, ActivationFunction.Type.GELU, VOCAB_SIZE,
                Map.of("LABEL_0", 0, "LABEL_1", 1, "LABEL_2", 2), 102, 101);
    }

    private static int[] inputIds() {
        int[] ids = new int[BATCH_SIZE * SEQ_LENGTH];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = (i * 7 + 3) % VOCAB_SIZE;
        }
        return ids;
    }

    private static int[] onesArray(int length) {
        int[] values = new int[length];
        java.util.Arrays.fill(values, 1);
        return values;
    }

    private static int[] zerosArray(int length) {
        return new int[length];
    }

    private static FloatBufferTensor matrix(int rows, int cols, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(matrixValue(row, col, seed), row, col);
            }
        }
        return tensor;
    }

    private static float[] expectedEmbeddingRow(int inputId, int tokenTypeId, int positionId) {
        float[] raw = new float[HIDDEN_SIZE];
        float sum = 0.0f;
        float sumSq = 0.0f;
        for (int col = 0; col < HIDDEN_SIZE; col++) {
            float value = matrixValue(inputId, col, 1)
                    + matrixValue(tokenTypeId, col, 2)
                    + matrixValue(positionId, col, 3);
            raw[col] = value;
            sum += value;
            sumSq += value * value;
        }
        float mean = sum / HIDDEN_SIZE;
        float variance = sumSq / HIDDEN_SIZE - mean * mean;
        float invStdDev = 1.0f / (float) Math.sqrt(variance + 1.0e-12f);
        for (int col = 0; col < HIDDEN_SIZE; col++) {
            raw[col] = (raw[col] - mean) * invStdDev;
        }
        return raw;
    }

    private static float matrixValue(int row, int col, int seed) {
        return ((row * 13 + col * 7 + seed) % 17 - 8) / 16.0f;
    }

    private static AbstractTensor tensorFromFormula(BertModel model, int rows, int cols, float offset) {
        AbstractTensor tensor = model.makeDenseTensor(rows, cols);
        float[][] values = formulaRows(rows, cols, offset);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(values[row][col], row, col);
            }
        }
        return tensor;
    }

    private static float[][] formulaRows(int rows, int cols, float offset) {
        float[][] values = new float[rows][cols];
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                values[row][col] = offset + row * 0.125f + col * 0.03125f;
            }
        }
        return values;
    }

    private static float[][] add(AbstractTensor left, float[][] right) {
        float[][] values = new float[(int) left.shape().first()][(int) left.shape().last()];
        for (int row = 0; row < values.length; row++) {
            for (int col = 0; col < values[row].length; col++) {
                values[row][col] = left.get(row, col) + right[row][col];
            }
        }
        return values;
    }

    private static float[][] add(float[][] left, float[][] right) {
        float[][] values = new float[left.length][left[0].length];
        for (int row = 0; row < values.length; row++) {
            for (int col = 0; col < values[row].length; col++) {
                values[row][col] = left[row][col] + right[row][col];
            }
        }
        return values;
    }

    private static float[][] layerNorm(float[][] input) {
        float[][] output = new float[input.length][input[0].length];
        for (int row = 0; row < input.length; row++) {
            float sum = 0.0f;
            float sumSq = 0.0f;
            for (int col = 0; col < input[row].length; col++) {
                sum += input[row][col];
                sumSq += input[row][col] * input[row][col];
            }
            float mean = sum / input[row].length;
            float variance = sumSq / input[row].length - mean * mean;
            float invStdDev = 1.0f / (float) Math.sqrt(variance + 1.0e-12f);
            for (int col = 0; col < input[row].length; col++) {
                output[row][col] = (input[row][col] - mean) * invStdDev;
            }
        }
        return output;
    }

    private static FloatBufferTensor ones(int rows, int cols) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(1.0f, row, col);
            }
        }
        return tensor;
    }

    private static FloatBufferTensor zeros(int rows, int cols) {
        return new FloatBufferTensor(rows, cols);
    }
}
