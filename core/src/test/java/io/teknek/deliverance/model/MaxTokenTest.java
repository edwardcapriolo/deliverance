package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.EmbedInput;
import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.generator.SampleOutput;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.guided.LogitsProcessor;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorInfo;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Queue;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MaxTokenTest {

    @Test
    public void maxTokens() {
        try (TinyReplayModel model = new TinyReplayModel()) {
            Response response = new GenerationEngine().generate(model,
                    new ReplayBackend(),
                    new ReplayTokenSampler(List.of(1, 2, 3, 4, 5, 6, 7, 8, 9)),
                    UUID.randomUUID(),
                    PromptContext.of("prompt"),
                    new GeneratorParameters()
                            .withNtokens(32)
                            .withMaxTokens(8)
                            .withTemperature(0.0f),
                    new DoNothingGenerateEvent());

            assertEquals(FinishReason.MAX_TOKENS, response.finishReason);
            assertEquals(8, response.generatedTokens.size());
            assertEquals(List.of(1, 2, 3, 4, 5, 6, 7, 8), response.generatedTokens);
            assertEquals("abcdefgh", response.responseText);
        }
    }

    private static final class ReplayTokenSampler implements TokenSampler {
        private final Queue<Integer> tokens;

        private ReplayTokenSampler(List<Integer> tokens) {
            this.tokens = new ArrayDeque<>(tokens);
        }

        @Override
        public SamplerReturn firstToken(GeneratorParameters parameters, GenerationEngine.Logits logits,
                GenerationEngine.PrefillOutput prefill, ResponseContext responseContext, Random random, float temperature,
                Optional<LogitsProcessor> logitsProcessor) {
            return nextReplayToken();
        }

        @Override
        public SamplerReturn nextToken(GeneratorParameters parameters, AbstractTensor output, AbstractTensor logits,
                ResponseContext responseContext, Random random, float temperature,
                Optional<LogitsProcessor> logitsProcessor) {
            return nextReplayToken();
        }

        private SamplerReturn nextReplayToken() {
            Integer token = tokens.poll();
            if (token == null) {
                throw new IllegalStateException("replay sampler exhausted");
            }
            return new SamplerReturn(token);
        }
    }

    private static final class ReplayBackend implements GenerationBackend {
        @Override
        public GenerationSession open(UUID sessionId, int[] promptTokens, GeneratorParameters parameters) {
            return new GenerationSession() {
                @Override
                public int prefixLength() {
                    return 0;
                }

                @Override
                public AbstractTensor prefill(GenerationCursor cursor) {
                    return new FloatBufferTensor(1, 1);
                }

                @Override
                public AbstractTensor decode(int tokenId, int position) {
                    return new FloatBufferTensor(1, 1);
                }

                @Override
                public void close() {
                }
            };
        }
    }

    private static final class TinyReplayModel extends AbstractModel {
        private static final Config CONFIG = new Config(32, 4, 16, 1, 1, 1, 1.0e-6f,
                16, 0, List.of(0), ActivationFunction.Type.SILU, null, Map.of());

        private TinyReplayModel() {
            super(InferenceType.OUTPUT_TO_TOKEN, CONFIG, new TinyWeightLoader(), null, DType.F32, DType.F32,
                    Optional.empty(), new ConfigurableTensorProvider(new NaiveTensorOperations()), new MetricRegistry(),
                    new ArrayQueueTensorAllocator(new MetricRegistry()), new KvBufferCacheSettings(true),
                    new DefaultToolCallParser(), new WrappedForkJoinPool(new ForkJoinPool(1)));
        }

        @Override
        protected long[] encodeText(String text) {
            return new long[] {11, 12, 13};
        }

        @Override
        public String decodeToken(int token) {
            return switch (token) {
                case 1 -> "a";
                case 2 -> "b";
                case 3 -> "c";
                case 4 -> "d";
                case 5 -> "e";
                case 6 -> "f";
                case 7 -> "g";
                case 8 -> "h";
                case 9 -> "i";
                default -> "";
            };
        }

        @Override
        public boolean isSpecialToken(int token) {
            return false;
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
