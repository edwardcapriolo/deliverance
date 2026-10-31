package io.teknek.deliverance.model;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.EmbedInput;
import io.teknek.deliverance.generator.SampleOutput;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.grace.EncodeOptions;
import io.teknek.deliverance.grace.Encoding;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;
import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class AbstractModelPromptConstructionTest {

    private static final String TEMPLATE_PROMPT = "<template-bos>hello";
    private static final String RAW_PROMPT = "hello";

    static Stream<Arguments> modelFamilies() {
        return Stream.of(
                Arguments.of("Gemma2", 2, new int[]{2, 101, 102}),
                Arguments.of("Gemma4", 2, new int[]{2, 201, 202}),
                Arguments.of("Qwen", 151643, new int[]{151643, 301, 302}),
                Arguments.of("Llama", 128000, new int[]{128000, 401, 402}));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("modelFamilies")
    void constructsPromptTokensWithoutDuplicateBos(String family, int bosToken, int[] templateEncoded) {
        PreTrainedTokenizer tokenizer = mock(PreTrainedTokenizer.class);
        when(tokenizer.encode(eq(TEMPLATE_PROMPT), any(EncodeOptions.class)))
                .thenReturn(encoding(templateEncoded));
        when(tokenizer.encode(eq(RAW_PROMPT), any(EncodeOptions.class)))
                .thenReturn(encoding(new int[]{templateEncoded[1], templateEncoded[2]}));

        TinyModel model = new TinyModel(bosToken, tokenizer);
        try {
            assertArrayEquals(templateEncoded,
                    model.constructPromptTokensForRuntime(TEMPLATE_PROMPT), family);
            assertArrayEquals(new int[]{bosToken, templateEncoded[1], templateEncoded[2]},
                    model.constructPromptTokensForRuntime(RAW_PROMPT), family);
        } finally {
            model.close();
        }
    }

    private static Encoding encoding(int[] inputIds) {
        return new Encoding(inputIds, new int[inputIds.length], new int[inputIds.length]);
    }

    private static final class TinyModel extends AbstractModel {
        private static final Config CONFIG = new Config(16, 4, 8, 2, 2, 1, 1.0e-6f,
                200_000, 0, java.util.List.of(0), ActivationFunction.Type.SILU, null, Map.of());

        TinyModel(int bosToken, PreTrainedTokenizer tokenizer) {
            super(InferenceType.OUTPUT_TO_TOKEN, withBosToken(bosToken), weightLoader(), tokenizer,
                    DType.F32, DType.F32, Optional.empty(),
                    new ConfigurableTensorProvider(new NaiveTensorOperations()), new MetricRegistry(),
                    new ArrayQueueTensorAllocator(new MetricRegistry()), new KvBufferCacheSettings(true),
                    new DefaultToolCallParser(), new WrappedForkJoinPool(new ForkJoinPool(1)),
                    new StaticTensorParallelContext(0, 1));
        }

        private static Config withBosToken(int bosToken) {
            return new Config(CONFIG.contextLength, CONFIG.embeddingLength, CONFIG.hiddenLength,
                    CONFIG.numberOfHeads, CONFIG.numberOfKeyValueHeads, CONFIG.numberOfLayers,
                    CONFIG.layerNormEps, CONFIG.vocabularySize, bosToken, CONFIG.eosTokens,
                    CONFIG.activationFunction, null, Map.of());
        }

        private static WeightLoader weightLoader() {
            WeightLoader loader = mock(WeightLoader.class);
            when(loader.getModelDType()).thenReturn(DType.F32);
            when(loader.metadata()).thenReturn(Map.of());
            when(loader.tensorInfoMap()).thenReturn(Map.of());
            return loader;
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
}
