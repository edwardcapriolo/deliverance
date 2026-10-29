package io.teknek.deliverance.embedding;

import com.fasterxml.jackson.databind.JsonNode;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForEmbeddings;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;

/** Emits one deterministic Java EmbeddingGemma 2 vector for direct Python comparison. */
public final class EmbeddingGemma2ReferenceMain {
    private static final String DEFAULT_TEXT = "This is a test document about machine learning";

    private EmbeddingGemma2ReferenceMain() {
    }

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        MetricRegistry metrics = new MetricRegistry();
        ArrayQueueTensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(options.poolSize));
             AbstractModel model = AutoModelForEmbeddings.newBuilder(options.modelDirectory)
                     .buildLocalEmbeddingModel()) {
            int[] tokenIds = model.getTokenizer().encode(options.text).inputIds();
            float[] embedding = model.embed(options.text, PoolingType.AVG);
            double norm = 0.0;
            for (float value : embedding) norm += (double) value * value;
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("text", options.text);
            result.put("tokenIds", tokenIds);
            result.put("embedding", embedding);
            result.put("dimension", embedding.length);
            result.put("l2Norm", Math.sqrt(norm));
            JsonUtils.om.writeValue(options.output.toFile(), result);
            System.out.println(JsonUtils.om.writeValueAsString(result));
        }
    }

    private record Options(Path modelDirectory, String text, Path output, int poolSize) {
        static Options parse(String[] args) throws Exception {
            Path model = null;
            String text = DEFAULT_TEXT;
            Path output = Path.of("core/target/embedding-gemma2-reference-java.json");
            int poolSize = Math.max(1, Runtime.getRuntime().availableProcessors());
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--model-dir" -> model = Path.of(args[++i]);
                    case "--text" -> text = args[++i];
                    case "--output" -> output = Path.of(args[++i]);
                    case "--pool-size" -> poolSize = Integer.parseInt(args[++i]);
                    default -> throw new IllegalArgumentException("Unknown argument " + args[i]);
                }
            }
            if (model == null) throw new IllegalArgumentException("--model-dir is required");
            Files.createDirectories(output.toAbsolutePath().getParent());
            return new Options(model, text, output, poolSize);
        }
    }
}
