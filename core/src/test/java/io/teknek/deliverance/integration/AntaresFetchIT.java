package io.teknek.deliverance.integration;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

import io.teknek.deliverance.model.GenerateEvent;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.RmsNorm;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.TensorInfo;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("large-model")
class AntaresFetchIT {

    @Test
    void antares1bMetadataMatchesGraniteMoeHybridDenseShape() throws IOException {
        ModelFetcher fetch = new ModelFetcher("fdtn-ai", "antares-1b");
        File modelRoot = fetch.maybeDownload();

        var config = JsonUtils.om.readTree(new File(modelRoot, "config.json"));
        assertEquals("granitemoehybrid", config.get("model_type").asText());
        assertEquals(40, config.get("num_hidden_layers").asInt());
        assertEquals(2048, config.get("hidden_size").asInt());
        assertEquals(4096, config.get("shared_intermediate_size").asInt());
        assertEquals(0, config.get("num_local_experts").asInt());
        assertEquals(0, config.get("num_experts_per_tok").asInt());
        config.get("layer_types").forEach(layerType -> assertEquals("attention", layerType.asText()));

        try (DefaultWeightLoader loader = new DefaultWeightLoader(modelRoot)) {
            Map<String, TensorInfo> tensors = loader.tensorInfoMap();
            assertTensor(tensors, "model.embed_tokens.weight", 100352, 2048);
            assertTensor(tensors, "model.layers.0.input_layernorm.weight", 2048);
            assertTensor(tensors, "model.layers.0.self_attn.q_proj.weight", 2048, 2048);
            assertTensor(tensors, "model.layers.0.self_attn.k_proj.weight", 512, 2048);
            assertTensor(tensors, "model.layers.0.self_attn.v_proj.weight", 512, 2048);
            assertTensor(tensors, "model.layers.0.self_attn.o_proj.weight", 2048, 2048);
            assertTensor(tensors, "model.layers.0.post_attention_layernorm.weight", 2048);
            assertTensor(tensors, "model.layers.0.shared_mlp.input_linear.weight", 8192, 2048);
            assertTensor(tensors, "model.layers.0.shared_mlp.output_linear.weight", 2048, 4096);
            assertTensor(tensors, "model.norm.weight", 2048);
            assertTensor(tensors, "lm_head.weight", 100352, 2048);
            assertTrue(tensors.keySet().stream().noneMatch(name -> name.contains("block_sparse_moe")),
                    "Antares-1B config has no active MoE experts");
        }
    }

    @Test
    void antares1bLoadsAndGenerates() {
        ModelFetcher fetch = new ModelFetcher("fdtn-ai", "antares-1b");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).buildLocalTransformerModel()) {
            PromptSupport.Builder prompt = model.promptSupport().orElseThrow().builder()
                    .addUserMessage("Return one Java file path.");

            Response response = model.generate(UUID.randomUUID(), prompt.build(),
                    new GeneratorParameters().withTemperature(0.3f).withTopP(1.0f).withMaxTokens(64),
                    new GenerateEvent() {
                        @Override
                        public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                            System.out.print(next + " " +nextCleaned);
                        }
                    });

            System.out.println("ANTARES_1B_SMOKE=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
            assertFalse(response.responseTextWithSpecialTokens.contains("•\n•\n•"));
            assertTrue(response.responseTextWithSpecialTokens.contains("/"));
        }
    }

    @Test
    void antares1bJq4LoadsAndGeneratesWhenCached() {
        ModelFetcher fetch = new ModelFetcher("edwardcapriolo", "antares-1b-JQ4");
        Assumptions.assumeTrue(fetch.pathForModel().toFile().isDirectory(),
                "Quantized Antares cache is not present: " + fetch.pathForModel());
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).buildLocalTransformerModel()) {
            PromptSupport.Builder prompt = model.promptSupport().orElseThrow().builder()
                    .addUserMessage("Return one Java file path.");

            Response response = model.generate(UUID.randomUUID(), prompt.build(),
                    new GeneratorParameters().withTemperature(0.3f).withTopP(1.0f).withMaxTokens(64),
                    new DoNothingGenerateEvent());

            System.out.println("ANTARES_1B_JQ4_SMOKE=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
            assertFalse(response.responseTextWithSpecialTokens.contains("•\n•\n•"));
            assertTrue(response.responseTextWithSpecialTokens.contains("/"));
        }
    }

    @Test
    void antares1bLogitsMatchTransformersReferenceExactly() {
        ModelFetcher fetch = new ModelFetcher("fdtn-ai", "antares-1b");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).buildLocalTransformerModel();
             DefaultWeightLoader weights = new DefaultWeightLoader(fetch.pathForModel().toFile());
             KvBufferCache.KvBuffer kvBuffer = model.newKvBuffer();
             AbstractTensor hidden = model.batchForward(new int[]{3, 4, 5, 6}, 0, kvBuffer);
             AbstractTensor outputNormWeights = weights.load("model.norm.weight");
             AbstractTensor outputWeights = weights.load("lm_head.weight")) {
            RmsNorm outputNorm = new RmsNorm(model, outputNormWeights, model.getMetricRegistry());
            try (AbstractTensor normalized = outputNorm.forward(hidden);
                 AbstractTensor logits = model.makeTensor(normalized.shape().first(), model.getConfig().vocabularySize)) {

                model.primaryTensorOperations().dotProductChunk(logits, normalized, outputWeights, 0,
                        model.getConfig().embeddingLength, 0, model.getConfig().vocabularySize);
                model.primaryTensorOperations().scale(1.0f / model.getConfig().logitMultiplier, logits, 0,
                        model.getConfig().vocabularySize);

                assertClose("mean logits by position",
                        new float[]{-4.943235874176025f, -1.1108887195587158f, -1.0443177223205566f,
                                -0.7825897932052612f},
                        new float[]{mean(logits, 0), mean(logits, 1), mean(logits, 2), mean(logits, 3)},
                        1.0e-3f);

                assertClose("last token logits slice",
                        new float[]{12.47355842590332f, 7.11851167678833f, 10.766719818115234f,
                                10.445048332214355f, 10.424182891845703f, 12.099858283996582f,
                                8.804206848144531f, 8.976678848266602f},
                        new float[]{logits.get(3, 0), logits.get(3, 1), logits.get(3, 2), logits.get(3, 3),
                                logits.get(3, 4), logits.get(3, 5), logits.get(3, 6), logits.get(3, 7)},
                        1.0e-3f);
            }
        }
    }

    private static void assertTensor(Map<String, TensorInfo> tensors, String name, int... shape) {
        TensorInfo tensor = tensors.get(name);
        assertTrue(tensor != null, "Missing tensor " + name + ". Available sample: "
                + tensors.keySet().stream().limit(20).toList());
        assertArrayEquals(shape, tensor.shape, name + " shape was " + Arrays.toString(tensor.shape));
    }

    private static float mean(AbstractTensor tensor, int row) {
        float sum = 0.0f;
        for (int col = 0; col < tensor.shape().last(); col++) {
            sum += tensor.get(row, col);
        }
        return sum / tensor.shape().last();
    }

    private static void assertClose(String label, float[] expected, float[] actual, float tolerance) {
        float maxDelta = 0.0f;
        for (int i = 0; i < expected.length; i++) {
            maxDelta = Math.max(maxDelta, Math.abs(expected[i] - actual[i]));
        }
        if (maxDelta > tolerance) {
            try (FloatBufferTensor expectedTensor = rowTensor(expected);
                 FloatBufferTensor actualTensor = rowTensor(actual)) {
                throw new AssertionError(label + " max delta " + maxDelta + " exceeded tolerance " + tolerance
                        + "\nexpected:\n" + TensorDisplayUtil.pretty2dDisplayAll(expectedTensor).trim()
                        + "\nactual:\n" + TensorDisplayUtil.pretty2dDisplayAll(actualTensor).trim());
            }
        }
    }

    private static FloatBufferTensor rowTensor(float[] values) {
        FloatBufferTensor tensor = new FloatBufferTensor(1, values.length);
        for (int i = 0; i < values.length; i++) {
            tensor.set(values[i], 0, i);
        }
        return tensor;
    }

}
