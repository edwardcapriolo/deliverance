package io.teknek.deliverance.embedding;

import com.fasterxml.jackson.databind.JsonNode;
import io.teknek.deliverance.JsonUtils;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;

/** Model-defined SentenceTransformers embedding recipe loaded from modules.json and Pooling config. */
public record SentenceTransformersEmbeddingRecipe(List<SentenceTransformersPooling.Mode> poolingModes,
                                                  boolean normalize,
                                                  OptionalInt maxSequenceLength) {
    public SentenceTransformersEmbeddingRecipe {
        poolingModes = List.copyOf(poolingModes);
        maxSequenceLength = maxSequenceLength == null ? OptionalInt.empty() : maxSequenceLength;
    }

    public static SentenceTransformersEmbeddingRecipe defaultMeanNormalize() {
        return new SentenceTransformersEmbeddingRecipe(List.of(SentenceTransformersPooling.Mode.MEAN), true,
                OptionalInt.empty());
    }

    public static SentenceTransformersEmbeddingRecipe fromModelRoot(Optional<Path> modelRoot) {
        return modelRoot.flatMap(SentenceTransformersEmbeddingRecipe::tryLoad).orElseGet(
                SentenceTransformersEmbeddingRecipe::defaultMeanNormalize);
    }

    private static Optional<SentenceTransformersEmbeddingRecipe> tryLoad(Path modelRoot) {
        Path modulesPath = modelRoot.resolve("modules.json");
        if (!Files.isRegularFile(modulesPath)) {
            return Optional.empty();
        }
        try {
            JsonNode modules = JsonUtils.om.readTree(modulesPath.toFile());
            List<SentenceTransformersPooling.Mode> modes = List.of(SentenceTransformersPooling.Mode.MEAN);
            boolean normalize = false;
            if (modules.isArray()) {
                for (JsonNode module : modules) {
                    String type = module.path("type").asText("");
                    String path = module.path("path").asText("");
                    if (type.endsWith(".Pooling")) {
                        modes = loadPoolingModes(modelRoot.resolve(path).resolve("config.json"));
                    } else if (type.endsWith(".Normalize")) {
                        normalize = true;
                    }
                }
            }
            return Optional.of(new SentenceTransformersEmbeddingRecipe(modes, normalize, loadMaxSequenceLength(modelRoot)));
        } catch (IOException e) {
            throw new UncheckedIOException("Unable to read SentenceTransformers metadata from " + modelRoot, e);
        }
    }

    private static OptionalInt loadMaxSequenceLength(Path modelRoot) throws IOException {
        Path config = modelRoot.resolve("sentence_bert_config.json");
        if (!Files.isRegularFile(config)) {
            return OptionalInt.empty();
        }
        JsonNode root = JsonUtils.om.readTree(config.toFile());
        JsonNode value = root.get("max_seq_length");
        if (value == null || !value.canConvertToInt()) {
            return OptionalInt.empty();
        }
        return OptionalInt.of(value.asInt());
    }

    private static List<SentenceTransformersPooling.Mode> loadPoolingModes(Path poolingConfig) throws IOException {
        if (!Files.isRegularFile(poolingConfig)) {
            return List.of(SentenceTransformersPooling.Mode.MEAN);
        }
        JsonNode config = JsonUtils.om.readTree(poolingConfig.toFile());
        List<SentenceTransformersPooling.Mode> modes = new ArrayList<>();
        if (config.path("pooling_mode_cls_token").asBoolean(false)) {
            modes.add(SentenceTransformersPooling.Mode.CLS);
        }
        if (config.path("pooling_mode_max_tokens").asBoolean(false)) {
            modes.add(SentenceTransformersPooling.Mode.MAX);
        }
        if (config.path("pooling_mode_mean_tokens").asBoolean(false)) {
            modes.add(SentenceTransformersPooling.Mode.MEAN);
        }
        if (config.path("pooling_mode_mean_sqrt_len_tokens").asBoolean(false)) {
            modes.add(SentenceTransformersPooling.Mode.MEAN_SQRT_LEN_TOKENS);
        }
        if (config.path("pooling_mode_weightedmean_tokens").asBoolean(false)) {
            modes.add(SentenceTransformersPooling.Mode.WEIGHTED_MEAN);
        }
        if (config.path("pooling_mode_lasttoken").asBoolean(false)) {
            modes.add(SentenceTransformersPooling.Mode.LAST_TOKEN);
        }
        return modes.isEmpty() ? List.of(SentenceTransformersPooling.Mode.MEAN) : modes;
    }
}
