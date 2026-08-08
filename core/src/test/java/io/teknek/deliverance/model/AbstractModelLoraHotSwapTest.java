package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.EmbedInput;
import io.teknek.deliverance.generator.SampleOutput;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.LoraAdapter;
import io.teknek.deliverance.safetensors.LoraAdapterConfig;
import io.teknek.deliverance.safetensors.LoraLayerDelta;
import io.teknek.deliverance.safetensors.SafeTensorWriter;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorInfo;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.Mockito;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests (no network) for {@link AbstractModel}'s LoRA runtime hot-swap registry and lifecycle
 * -- registration, activation, clearing, the tensor-parallel non-goal guard, and the
 * {@link AbstractModel#close()} resource fix. See step 4 plan Section 8's non-goal-guard and
 * close()-resource-test items.
 *
 * <p>Uses a minimal real {@link AbstractModel} subclass ({@code TinyModel}, modeled on {@code
 * GuidedChoiceLogitsProcessorTest.TinyChoiceModel}) rather than a bare Mockito mock: {@code
 * registerLoraAdapter}/{@code setActiveAdapter} read real instance fields ({@code
 * tensorParallelContext}, {@code registeredLoraAdapters}) that a constructor-bypassing mock would
 * leave {@code null}.</p>
 */
public class AbstractModelLoraHotSwapTest {

    private static final String Q_PROJ = "model.layers.0.self_attn.q_proj.weight";

    @TempDir
    Path tempDir;

    @Test
    void registerLoraAdapterThrowsWhenFamilyDoesNotSupportHotSwap() throws IOException {
        TinyModel model = new TinyModel(false, new StaticTensorParallelContext(0, 1), new TinyWeightLoader());
        try (LoraAdapter adapter = buildSyntheticAdapter()) {
            assertThrows(UnsupportedOperationException.class, () -> model.registerLoraAdapter("a", adapter));
        } finally {
            model.close();
        }
    }

    @Test
    void registerSetActiveClearAndUnregisterLifecycleWorks() throws IOException {
        TinyModel model = new TinyModel(true, new StaticTensorParallelContext(0, 1), new TinyWeightLoader());
        try {
            LoraAdapter adapter = buildSyntheticAdapter();
            model.registerLoraAdapter("a", adapter);

            assertTrue(model.activeLoraDeltaFor(Q_PROJ).isEmpty(), "no adapter active yet");

            model.setActiveAdapter("a");
            Optional<LoraLayerDelta> delta = model.activeLoraDeltaFor(Q_PROJ);
            assertTrue(delta.isPresent());
            assertTrue(model.activeLoraDeltaFor("model.layers.0.self_attn.k_proj.weight").isEmpty(),
                    "adapter doesn't target k_proj");

            assertThrows(IllegalStateException.class, () -> model.unregisterLoraAdapter("a"),
                    "cannot unregister the currently active adapter");

            model.clearActiveAdapter();
            assertTrue(model.activeLoraDeltaFor(Q_PROJ).isEmpty(), "clearing must stop applying the delta");

            model.unregisterLoraAdapter("a"); // fine now that it's no longer active
            assertThrows(IllegalArgumentException.class, () -> model.setActiveAdapter("a"),
                    "adapter is no longer registered");
        } finally {
            model.close();
        }
    }

    @Test
    void setActiveAdapterThrowsUnderTensorParallel() throws IOException {
        TinyModel model = new TinyModel(true, new StaticTensorParallelContext(0, 2), new TinyWeightLoader());
        try (LoraAdapter adapter = buildSyntheticAdapter()) {
            model.registerLoraAdapter("a", adapter);
            assertThrows(UnsupportedOperationException.class, () -> model.setActiveAdapter("a"));
        } finally {
            model.close();
        }
    }

    @Test
    void closeClosesRegisteredAdaptersAndTheBaseWeightLoader() throws IOException {
        TinyWeightLoader weightLoader = new TinyWeightLoader();
        TinyModel model = new TinyModel(true, new StaticTensorParallelContext(0, 1), weightLoader);
        LoraAdapter adapter = Mockito.spy(buildSyntheticAdapter());
        model.registerLoraAdapter("a", adapter);

        model.close();

        assertTrue(weightLoader.closed, "AbstractModel.close() must close the base WeightLoader");
        Mockito.verify(adapter).close();
    }

    private LoraAdapter buildSyntheticAdapter() throws IOException {
        Path adapterDir = Files.createTempDirectory(tempDir, "adapter");
        Files.writeString(adapterDir.resolve(LoraAdapterConfig.FILE_NAME),
                "{\"r\": 4, \"lora_alpha\": 8.0, \"target_modules\": [\"q_proj\"]}");
        // LoraTensorNames is package-private to io.teknek.deliverance.safetensors; its documented
        // convention ("base_model.model." + base-without-".weight" + ".lora_A/B.weight") is
        // inlined here rather than duplicated as a public helper just for this test.
        String withoutWeight = Q_PROJ.substring(0, Q_PROJ.length() - ".weight".length());
        String loraAName = "base_model.model." + withoutWeight + ".lora_A.weight";
        String loraBName = "base_model.model." + withoutWeight + ".lora_B.weight";
        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put(loraAName, new FloatBufferTensor(4, 4));
        tensors.put(loraBName, new FloatBufferTensor(4, 4));
        SafeTensorWriter.write(adapterDir.resolve(LoraAdapter.SAFETENSORS_FILE_NAME), Map.of(), tensors);
        return LoraAdapter.load(adapterDir.toFile());
    }

    private static final class TinyModel extends AbstractModel {
        // numberOfHeads=numberOfKeyValueHeads=2, embeddingLength=4 (headSize=2), hiddenLength=8:
        // divides evenly for tensor-parallel size 2, needed by the TP-guard test.
        private static final Config CONFIG = new Config(16, 4, 8, 2, 2, 1, 1.0e-6f,
                10, 0, List.of(0), ActivationFunction.Type.SILU, null, Map.of());
        private final boolean hotSwapSupported;

        TinyModel(boolean hotSwapSupported, TensorParallelContext tensorParallelContext, WeightLoader weightLoader) {
            super(InferenceType.OUTPUT_TO_TOKEN, CONFIG, weightLoader, null, DType.F32, DType.F32,
                    Optional.empty(), new ConfigurableTensorProvider(new NaiveTensorOperations()), new MetricRegistry(),
                    new ArrayQueueTensorAllocator(new MetricRegistry()), new KvBufferCacheSettings(true),
                    new DefaultToolCallParser(), new WrappedForkJoinPool(new ForkJoinPool(1)), tensorParallelContext);
            this.hotSwapSupported = hotSwapSupported;
        }

        @Override
        protected boolean supportsLoraHotSwap() {
            return hotSwapSupported;
        }

        @Override
        protected EmbedInput loadInputWeights() {
            return null;
        }

        @Override
        protected SampleOutput loadOutputWeights() {
            return null;
        }

        @Override
        protected TransformerBlock[] loadTransformerBlockWeights() {
            return new TransformerBlock[0];
        }
    }

    private static final class TinyWeightLoader implements WeightLoader {
        private boolean closed;

        @Override
        public Map<String, String> metadata() {
            return Map.of();
        }

        @Override
        public Map<String, TensorInfo> tensorInfoMap() {
            return Map.of();
        }

        @Override
        public DType getModelDType() {
            return DType.F32;
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}
