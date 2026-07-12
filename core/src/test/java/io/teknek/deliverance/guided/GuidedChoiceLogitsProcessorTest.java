package io.teknek.deliverance.guided;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.EmbedInput;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.SampleOutput;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ResponseContext;
import io.teknek.deliverance.safetensors.Config;
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

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GuidedChoiceLogitsProcessorTest {

    @Test
    void masksTokensThatDoNotContinueGuidedChoicePrefix() {
        TinyChoiceModel model = new TinyChoiceModel();
        try (AbstractTensor logits = new FloatBufferTensor(1, 5)) {
            logits.set(10.0f, 0, 1); // cat
            logits.set(20.0f, 0, 2); // dog
            logits.set(30.0f, 0, 3); // ma, completing dogma after dog
            logits.set(40.0f, 0, 4); // zebra

            ResponseContext responseContext = new ResponseContext(model);

            LogitsProcessor processor = LogitsProcessorFactory.create(model,
                    new GeneratorParameters().withGuidedChoice(List.of("dogma"))).orElseThrow();
            processor.accept(2, responseContext);
            processor.process(logits, responseContext);

            assertEquals(Float.NEGATIVE_INFINITY, logits.get(0, 1));
            assertEquals(Float.NEGATIVE_INFINITY, logits.get(0, 2));
            assertEquals(30.0f, logits.get(0, 3));
            assertEquals(Float.NEGATIVE_INFINITY, logits.get(0, 4));
        } finally {
            model.close();
        }
    }

    @Test
    void factoryCompilesGuidedChoiceParameters() {
        TinyChoiceModel model = new TinyChoiceModel();
        try {
            Optional<LogitsProcessor> processor = LogitsProcessorFactory.create(model,
                    new GeneratorParameters().withGuidedChoice(List.of("dogma")));
            assertTrue(processor.isPresent());
        } finally {
            model.close();
        }
    }

    @Test
    void masksTokensThatDoNotContinueGuidedRegex() {
        TinyChoiceModel model = new TinyChoiceModel();
        try (AbstractTensor logits = new FloatBufferTensor(1, 5)) {
            logits.set(10.0f, 0, 1); // cat
            logits.set(20.0f, 0, 2); // dog
            logits.set(30.0f, 0, 3); // ma
            logits.set(40.0f, 0, 4); // zebra

            ResponseContext responseContext = new ResponseContext(model);

            LogitsProcessor processor = LogitsProcessorFactory.create(model,
                    new GeneratorParameters().withGuidedRegex("dogma")).orElseThrow();
            processor.accept(2, responseContext);
            processor.process(logits, responseContext);

            assertEquals(Float.NEGATIVE_INFINITY, logits.get(0, 1));
            assertEquals(Float.NEGATIVE_INFINITY, logits.get(0, 2));
            assertEquals(30.0f, logits.get(0, 3));
            assertEquals(Float.NEGATIVE_INFINITY, logits.get(0, 4));
        } finally {
            model.close();
        }
    }

    @Test
    void rejectsConflictingGuidanceModes() {
        TinyChoiceModel model = new TinyChoiceModel();
        try {
            GeneratorParameters parameters = new GeneratorParameters()
                    .withGuidedChoice(List.of("dogma"))
                    .withGuidedRegex("dogma");

            assertThrows(IllegalArgumentException.class, () -> LogitsProcessorFactory.create(model, parameters));
        } finally {
            model.close();
        }
    }

    private static final class TinyChoiceModel extends AbstractModel {
        private static final Config CONFIG = new Config(16, 4, 8, 1, 1, 1, 1.0e-6f,
                5, 0, List.of(0), ActivationFunction.Type.SILU, null, Map.of());

        private TinyChoiceModel() {
            super(InferenceType.OUTPUT_TO_TOKEN, CONFIG, new TinyWeightLoader(), null, DType.F32, DType.F32,
                    Optional.empty(), new ConfigurableTensorProvider(new NaiveTensorOperations()), new MetricRegistry(),
                    new ArrayQueueTensorAllocator(new MetricRegistry()), new KvBufferCacheSettings(true),
                    new DefaultToolCallParser(), new WrappedForkJoinPool(new ForkJoinPool(1)));
        }

        @Override
        protected long[] encodeText(String text) {
            return switch (text) {
                case "cat" -> new long[] {1};
                case "dog" -> new long[] {2};
                case "dogma" -> new long[] {2, 3};
                case "zebra" -> new long[] {4};
                default -> throw new IllegalArgumentException("Unexpected text: " + text);
            };
        }

        @Override
        public String decodeToken(int token) {
            return switch (token) {
                case 1 -> "cat";
                case 2 -> "dog";
                case 3 -> "ma";
                case 4 -> "zebra";
                default -> "";
            };
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
