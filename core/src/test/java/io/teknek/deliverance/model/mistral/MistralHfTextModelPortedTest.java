package io.teknek.deliverance.model.mistral;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.hf.HfConfigTesterMixinPort;
import io.teknek.deliverance.model.hf.HfModelTesterMixinPort;
import io.teknek.deliverance.model.llama.LlamaHfTextModelPortedTest;
import io.teknek.deliverance.model.tensorparallel.SingleRankTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MistralHfTextModelPortedTest implements HfConfigTesterMixinPort, HfModelTesterMixinPort {
    @TempDir
    Path tempDir;

    @Override
    public Path hfTestTempDir() {
        return tempDir;
    }

    @Override
    public Path writeTinyCheckpoint(String name, int seed) {
        return LlamaHfTextModelPortedTest.writeTinyCheckpoint(tempDir.resolve(name),
                LlamaHfTextModelPortedTest.tinyConfig(), seed);
    }

    @Override
    public AbstractModel loadTinyModel(Path modelDir) {
        try {
            Config config = JsonUtils.om.readValue(modelDir.resolve("config.json").toFile(),
                    io.teknek.deliverance.model.llama.LlamaConfig.class);
            MetricRegistry metrics = new MetricRegistry();
            TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
            WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
            MistralModel model = new MistralModel(AbstractModel.InferenceType.FULL_GENERATION, config,
                    new DefaultWeightLoader(modelDir.toFile()), Mockito.mock(PreTrainedTokenizer.class),
                    DType.F32, DType.F32, Optional.empty(), metrics, new KvBufferCacheSettings(true),
                    new DefaultToolCallParser(), pool, new StaticTensorParallelContext(0, 1),
                    new SingleRankTensorParallelCollectives(), Optional.empty(), new Lighter(metrics, pool.getUnderlying()));
            model.init();
            return model;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Config loadTinyConfig(Path modelDir) {
        try {
            return JsonUtils.om.readValue(modelDir.resolve("config.json").toFile(),
                    io.teknek.deliverance.model.llama.LlamaConfig.class);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Config roundTripConfig(Config config) throws Exception {
        return config;
    }

    @Override
    @Disabled("Mistral uses KVCache2; this generic check targets the legacy KvBuffer API")
    public void hfModelTesterPastKeyValuesFormatEquivalent() {
    }

    @Override
    public int[] hfSampleTokenIds() {
        return new int[]{3, 4, 5, 6};
    }

    @Override
    public AbstractTensor makeInputsEmbeds(int rows, int embeddingLength, int seed) {
        return null;
    }

    @Test
    public void slidingWindowHandlesPrefillBeyondWindow() throws Exception {
        Path modelDir = writeTinyCheckpoint("mistral-sliding-window", 22_001);
        @SuppressWarnings("unchecked")
        Map<String, Object> config = JsonUtils.om.readValue(
                modelDir.resolve("config.json").toFile(), LinkedHashMap.class);
        config.put("sliding_window", 4);
        JsonUtils.om.writeValue(modelDir.resolve("config.json").toFile(), config);

        AbstractModel model = loadTinyModel(modelDir);
        try (AbstractTensor output = model.batchForward(new int[]{3, 4, 5, 6, 7, 8, 9, 10}, 0)) {
            assertEquals(8, output.shape().first());
            assertFiniteTensor(output);
        } finally {
            closeModel(model);
        }
    }
}
