package io.teknek.deliverance.model;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.EmbedInput;
import io.teknek.deliverance.generator.SampleOutput;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorInfo;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.tensor.kv.KvCacheManager;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor.kv.KvWriteCursor;
import io.teknek.deliverance.tensor.kv.CacheExecutionMode;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ForkJoinPool;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class LocalGenerationBackendTest {
    private final MetricRegistry metricRegistry = new MetricRegistry();
    private final TensorAllocator allocator = new ArrayQueueTensorAllocator(metricRegistry);

    @Test
    void kvCache2SharedBlockModeUsesSharedLookupAndAdmission() {
        AbstractModel model = mock(AbstractModel.class);
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(2)
                .withPrefixCacheMode(KvBufferCacheSettings.PrefixCacheMode.SHARED_BLOCKS);
        KvCacheManager manager = new KvCacheManager(1, 8, 4, DType.F32, settings, allocator, metricRegistry);
        KvCacheSession kvSession = manager.openSession();
        int[] promptTokens = {1, 2, 3, 4};
        AbstractTensor prefillResult = allocator.getDirty(DType.F32, TensorShape.of(1, 4));
        when(model.usesKvCache2Generation()).thenReturn(true);
        when(model.prefixCacheMode()).thenReturn(KvBufferCacheSettings.PrefixCacheMode.SHARED_BLOCKS);
        when(model.activeLoraAdapterId()).thenReturn(Optional.empty());
        when(model.newKvCacheSession()).thenReturn(kvSession);
        when(model.restoreSharedPrefixToKvSession(same(promptTokens), eq(Optional.of("")), same(kvSession))).thenReturn(2);
        when(model.batchForward(any(int[].class), anyInt(), same(kvSession))).thenReturn(prefillResult);

        LocalGenerationBackend backend = new LocalGenerationBackend(model);

        try (GenerationBackend.GenerationSession session = backend.open(UUID.randomUUID(), promptTokens,
                new GeneratorParameters())) {
            assertEquals(2, session.prefixLength());
            AbstractTensor actual = session.prefill(GenerationCursor.from(promptTokens, session.prefixLength()));
            assertEquals(prefillResult, actual);
        }

        verify(model).restoreSharedPrefixToKvSession(same(promptTokens), eq(Optional.of("")), same(kvSession));
        verify(model).batchForward(assertIntArrayEquals(new int[] {3, 4}), eq(2), same(kvSession));
        verify(model).storeSharedPrefixFromKvSession(same(promptTokens), same(kvSession), eq(Optional.of("")));
        verify(model, never()).kvPrefixSnapshotCache();
    }

    @Test
    void abstractModelStoresAndRestoresSharedKvCache2Blocks() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(2)
                .withPrefixCacheMode(KvBufferCacheSettings.PrefixCacheMode.SHARED_BLOCKS);
        try (TinyKvCache2Model model = new TinyKvCache2Model(settings)) {
            int[] promptTokens = {1, 2, 3, 4};
            try (KvCacheSession first = model.newKvCacheSession()) {
                writePrompt(first, model.getTensorAllocator());
                model.storeSharedPrefixFromKvSession(promptTokens, first, Optional.of(""));
                assertEquals(2, model.kvBlockManager().residentBlockCount());
            }

            try (KvCacheSession second = model.newKvCacheSession()) {
                int restored = model.restoreSharedPrefixToKvSession(promptTokens, Optional.of(""), second);

                assertEquals(4, restored);
                assertEquals(4, second.length());
                try (AbstractTensor key = second.keyRowCopy(0, 3);
                     AbstractTensor value = second.valueRowCopy(0, 3)) {
                    assertEquals(41.0f, key.get(0, 0), 0.0f);
                    assertEquals(42.0f, value.get(0, 0), 0.0f);
                }
            }
        }
    }

    private static void writePrompt(KvCacheSession session, TensorAllocator allocator) {
        try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
            for (int position = 0; position < 4; position++) {
                try (AbstractTensor key = row(allocator, (position + 1) * 10.0f + 1.0f);
                     AbstractTensor value = row(allocator, (position + 1) * 10.0f + 2.0f)) {
                    writer.write(0, position, key, value);
                }
            }
            writer.advanceLength(4);
        }
    }

    private static AbstractTensor row(TensorAllocator allocator, float first) {
        AbstractTensor tensor = allocator.getDirty(DType.F32, TensorShape.of(1, 4));
        for (int i = 0; i < 4; i++) {
            tensor.set(first + i, 0, i);
        }
        return tensor;
    }

    private static int[] assertIntArrayEquals(int[] expected) {
        return org.mockito.ArgumentMatchers.argThat(actual -> {
            assertArrayEquals(expected, actual);
            return true;
        });
    }

    private static final class TinyKvCache2Model extends AbstractModel {
        private static final Config CONFIG = new Config(8, 4, 8, 1, 1, 1, 1.0e-6f,
                16, 0, List.of(0), ActivationFunction.Type.SILU, 10_000.0, Map.of());

        private TinyKvCache2Model(KvBufferCacheSettings settings) {
            super(InferenceType.OUTPUT_TO_TOKEN, CONFIG, new TinyWeightLoader(), null, DType.F32, DType.F32,
                    Optional.empty(), new ConfigurableTensorProvider(new NaiveTensorOperations()), new MetricRegistry(),
                    new ArrayQueueTensorAllocator(new MetricRegistry()), settings, new DefaultToolCallParser(),
                    new WrappedForkJoinPool(new ForkJoinPool(1)));
        }

        @Override
        public boolean usesKvCache2Generation() {
            return true;
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
        }
    }
}
