package io.teknek.deliverance.integration;

import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.granitemoehybrid.GraniteMoeHybridConfig;
import io.teknek.deliverance.model.granitemoehybrid.GraniteMoeHybridModel;
import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.TensorInfo;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("large-model")
class GraniteTinyFetchIT {

    @Test
    void granite40HTinyDownloadsAndExposesHybridMetadata() throws IOException {
        ModelFetcher fetch = new ModelFetcher("ibm-granite", "granite-4.0-h-tiny");
        File modelRoot = fetch.maybeDownload();

        var config = JsonUtils.om.readTree(new File(modelRoot, "config.json"));
        assertEquals("granitemoehybrid", config.get("model_type").asText());
        assertEquals(64, config.get("num_local_experts").asInt());
        assertEquals(6, config.get("num_experts_per_tok").asInt());
        assertTrue(config.get("layer_types").toString().contains("mamba"));

        try (DefaultWeightLoader loader = new DefaultWeightLoader(modelRoot)) {
            Map<String, TensorInfo> tensors = loader.tensorInfoMap();
            assertTrue(tensors.containsKey("model.embed_tokens.weight"));
            assertTrue(tensors.containsKey("model.layers.0.input_layernorm.weight"));
            assertTrue(tensors.containsKey("model.norm.weight"));
            assertFalse(tensors.keySet().stream().noneMatch(name -> name.contains("block_sparse_moe")),
                    "granite-4.0-h-tiny should include MoE weights");
        }
    }

    @Test
    void granite40HTinyBuildsHybridModel() {
        ModelFetcher fetch = new ModelFetcher("ibm-granite", "granite-4.0-h-tiny");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).buildLocalTransformerModel()) {
            assertTrue(model instanceof GraniteMoeHybridModel);
            GraniteMoeHybridConfig config = (GraniteMoeHybridConfig) model.getConfig();
            assertEquals(40, config.numberOfLayers);
            assertEquals(64, config.numLocalExperts);
            assertEquals(6, config.numExpertsPerToken);
            assertTrue(config.layerTypes.contains("mamba"));
            assertTrue(config.layerTypes.contains("attention"));
        }
    }

    @Test
    void granite40HTinySingleTokenForwardIsFinite() {
        ModelFetcher fetch = new ModelFetcher("ibm-granite", "granite-4.0-h-tiny");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).buildLocalTransformerModel();
             KvBufferCache.KvBuffer kv = model.newKvBuffer();
             AbstractTensor output = model.batchForward(new int[]{100257}, 0, kv)) {
            assertEquals(1, output.shape().first());
            assertEquals(model.getConfig().embeddingLength, output.shape().last());
            for (int col = 0; col < output.shape().last(); col++) {
                assertTrue(Float.isFinite(output.get(0, col)), "non-finite hidden value at column " + col);
            }
        }
    }

    @Test
    void granite40HTinyGeneratesShortSmoke() {
        ModelFetcher fetch = new ModelFetcher("ibm-granite", "granite-4.0-h-tiny");
        assertGraniteGeneratesShortSmoke(fetch, "GRANITE_TINY_SMOKE");
    }

    @Test
    void granite40HTinyJq4GeneratesShortSmokeWhenCached() {
        ModelFetcher fetch = new ModelFetcher("ibm-granite", "granite-4.0-h-tiny-JQ4");
        Assumptions.assumeTrue(fetch.pathForModel().toFile().isDirectory(),
                "Quantized Granite tiny cache is not present: " + fetch.pathForModel());
        assertGraniteGeneratesShortSmoke(fetch, "GRANITE_TINY_JQ4_SMOKE");
    }

    private static void assertGraniteGeneratesShortSmoke(ModelFetcher fetch, String label) {
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).buildLocalTransformerModel()) {
            PromptSupport.Builder prompt = model.promptSupport().orElseThrow().builder()
                    .addUserMessage("In one short sentence, say what a shell command does.");
            Response response = model.generate(UUID.randomUUID(), prompt.build(),
                    new GeneratorParameters().withTemperature(0.1f).withTopP(1.0f).withMaxTokens(8),
                    new DoNothingGenerateEvent());
            System.out.println(label + "=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
        }
    }
}
