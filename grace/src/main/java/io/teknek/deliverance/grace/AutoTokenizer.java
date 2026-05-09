package io.teknek.deliverance.grace;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.teknek.deliverance.grace.bert.BertTokenizer;
import io.teknek.deliverance.grace.models.TokenizerConfig;

import java.io.File;
import java.io.IOException;
import java.math.BigInteger;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class AutoTokenizer {

    private static final ConcurrentHashMap<String, Class<PreTrainedTokenizerBase>> registry = new ConcurrentHashMap<>();

    static {
    }

    public static PreTrainedTokenizer fromPretrained(OwnerNameOrPath ownerNameOrPath,
                                                     Optional<String> tokenizerType,
                                                     Optional<List<Object>> inputs,
                                                     Map<String, ?> tokenizerInitArgs) {
        File path = resolveModelDirectory(ownerNameOrPath);
        ObjectMapper objectMapper = new ObjectMapper();

        try {
            JsonNode tokenizerDocument = objectMapper.readTree(new File(path, "tokenizer.json"));
            JsonNode tokenizerConfigDocument = readOptionalJson(objectMapper, new File(path, "tokenizer_config.json"));
            JsonNode specialTokensMapDocument = readOptionalJson(objectMapper, new File(path, "special_tokens_map.json"));

            Map<String, Integer> vocab = readVocab(path, tokenizerDocument, objectMapper);
            SortedMap<Integer, AddedToken> addedTokenMap = computeSortedTokenMap(tokenizerDocument.path("added_tokens"));
            if (addedTokenMap.isEmpty()) {
                addedTokenMap = computeSortedTokenMap(tokenizerConfigDocument == null ? null : tokenizerConfigDocument.path("added_tokens_decoder"));
            }

            TokenizerConfig tokenizerConfig = parseTokenizerConfig(path, tokenizerConfigDocument, specialTokensMapDocument);
            BytePairEncodingModel bytePairEncodingModel = parseBytePairEncodingModel(path, tokenizerDocument, vocab);
            String tokenizerClass = tokenizerConfig.tokenizerClass() != null
                    ? tokenizerConfig.tokenizerClass()
                    : tokenizerType.orElse(null);

            if ("Qwen2Tokenizer".equals(tokenizerClass)) {
                return new Quen2Tokenizer(new LinkedHashMap<>(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), inputs,
                        vocab, addedTokenMap, tokenizerConfig, bytePairEncodingModel);
            }
            if (isGemmaLikeTokenizer(tokenizerClass, tokenizerDocument)) {
                return new GemmaTokenizer(new LinkedHashMap<>(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), inputs,
                        vocab, addedTokenMap, tokenizerConfig, bytePairEncodingModel);
            }
            if (isByteLevelBpeTokenizer(tokenizerClass, tokenizerDocument)) {
                return new ByteLevelBpeTokenizer(new LinkedHashMap<>(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), inputs,
                        vocab, addedTokenMap, tokenizerConfig, bytePairEncodingModel);
            }
            if ("BertTokenizer".equals(tokenizerClass)) {
                return new BertTokenizer(new LinkedHashMap<>(), Optional.empty(), Optional.empty(), Optional.empty(),
                        Optional.empty(), Optional.empty(), Optional.empty(), inputs,
                        vocab, addedTokenMap, tokenizerConfig, bytePairEncodingModel);
            }

            throw new io.teknek.dysfx.exception.UnreachableException("Could not find implementation " + tokenizerClass);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static PreTrainedTokenizer fromPretrained(OwnerNameOrPath ownerNameOrPath) {
        return fromPretrained(ownerNameOrPath, Optional.empty(), Optional.empty(), Collections.emptyMap());
    }

    public static PreTrainedTokenizer fromPretrained(Path modelPath) {
        return fromPretrained(OwnerNameOrPath.fromPath(modelPath));
    }

    public static SortedMap<Integer, AddedToken> computeSortedTokenMap(JsonNode addedTokens) {
        SortedMap<Integer, AddedToken> addedTokenMap = new TreeMap<>();
        if (addedTokens == null || addedTokens.isMissingNode() || addedTokens.isNull()) {
            return addedTokenMap;
        }

        if (addedTokens.isArray()) {
            for (JsonNode addedToken : addedTokens) {
                addTokenNode(addedTokenMap, addedToken.get("id").asInt(), addedToken);
            }
            return addedTokenMap;
        }

        if (addedTokens.isObject()) {
            addedTokens.fields().forEachRemaining(entry -> addTokenNode(addedTokenMap, Integer.parseInt(entry.getKey()), entry.getValue()));
        }
        return addedTokenMap;
    }

    private static void addTokenNode(SortedMap<Integer, AddedToken> addedTokenMap, int id, JsonNode addedToken) {
        String content = readTokenValue(addedToken);
        boolean singleWord = readBoolean(addedToken, "single_word", false);
        boolean lstrip = readBoolean(addedToken, "lstrip", false);
        boolean rstrip = readBoolean(addedToken, "rstrip", false);
        boolean special = readBoolean(addedToken, "special", false);
        JsonNode normalizedNode = addedToken.get("normalized");
        AddedToken token = new AddedToken(content, singleWord, lstrip, rstrip, special,
                normalizedNode == null || normalizedNode.isNull() ? null : normalizedNode.asBoolean());
        addedTokenMap.put(id, token);
    }

    private static File resolveModelDirectory(OwnerNameOrPath ownerNameOrPath) {
        if (ownerNameOrPath.ownerName != null) {
            TokenizerModelFetcher fetcher = new TokenizerModelFetcher(ownerNameOrPath.ownerName.owner, ownerNameOrPath.ownerName.name);
            return fetcher.maybeDownload();
        }
        if (ownerNameOrPath.modelPath != null) {
            return ownerNameOrPath.modelPath.path.toFile();
        }
        throw new IllegalArgumentException("OwnerNameOrPath must contain either an owner/name pair or a local path");
    }

    private static Map<String, Integer> readVocab(File path, JsonNode tokenizerDocument, ObjectMapper objectMapper) throws IOException {
        File vocabFile = new File(path, "vocab.json");
        if (vocabFile.exists()) {
            return objectMapper.readValue(vocabFile, new TypeReference<Map<String, Integer>>() {
            });
        }

        JsonNode modelVocab = tokenizerDocument.path("model").path("vocab");
        if (modelVocab.isObject()) {
            return objectMapper.convertValue(modelVocab, new TypeReference<Map<String, Integer>>() {
            });
        }
        return Map.of();
    }

    private static TokenizerConfig parseTokenizerConfig(File path, JsonNode tokenizerConfigDocument, JsonNode specialTokensMapDocument) {
        JsonNode mergedSpecialSource = specialTokensMapDocument == null ? tokenizerConfigDocument : specialTokensMapDocument;
        Map<String, String> specialTokens = new LinkedHashMap<>();
        for (String attribute : PreTrainedTokenizerBase.SPECIAL_TOKEN_ATTRIBUTES) {
            String token = readTokenValue(mergedSpecialSource == null ? null : mergedSpecialSource.get(attribute));
            if (token == null && tokenizerConfigDocument != null) {
                token = readTokenValue(tokenizerConfigDocument.get(attribute));
            }
            if (token != null) {
                specialTokens.put(attribute, token);
            }
        }

        List<String> additionalSpecialTokens = readStringList(mergedSpecialSource == null ? null : mergedSpecialSource.get("additional_special_tokens"));
        if (additionalSpecialTokens.isEmpty() && tokenizerConfigDocument != null) {
            additionalSpecialTokens = readStringList(tokenizerConfigDocument.get("additional_special_tokens"));
        }

        String chatTemplate = readText(tokenizerConfigDocument, "chat_template");
        if (chatTemplate == null) {
            File chatTemplateFile = new File(path, "chat_template.jinja");
            if (chatTemplateFile.exists()) {
                try {
                    chatTemplate = Files.readString(chatTemplateFile.toPath());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }

        return new TokenizerConfig(
                chatTemplate,
                readText(tokenizerConfigDocument, "tokenizer_class"),
                readBoolean(tokenizerConfigDocument, "clean_up_tokenization_spaces", false),
                readBoolean(tokenizerConfigDocument, "split_special_tokens", false),
                readBigInteger(tokenizerConfigDocument, "model_max_length", TokenizerUtils.VERY_LARGE_INTEGER),
                parsePaddingSide(readText(tokenizerConfigDocument, "padding_side")),
                parseTruncationSide(readText(tokenizerConfigDocument, "truncation_side")),
                specialTokens,
                additionalSpecialTokens);
    }

    private static BytePairEncodingModel parseBytePairEncodingModel(File path, JsonNode tokenizerDocument, Map<String, Integer> vocab) throws IOException {
        JsonNode modelNode = tokenizerDocument.path("model");
        if (!"BPE".equals(modelNode.path("type").asText())) {
            return null;
        }

        List<String> merges = readMerges(modelNode.get("merges"));
        if (merges.isEmpty()) {
            File mergesFile = new File(path, "merges.txt");
            if (mergesFile.exists()) {
                merges = Files.readAllLines(mergesFile.toPath()).stream()
                        .filter(line -> !line.isBlank() && !line.startsWith("#"))
                        .toList();
            }
        }

        return new BytePairEncodingModel(
                vocab,
                BytePairEncodingModel.fromMerges(merges),
                parsePreTokenizer(tokenizerDocument.path("pre_tokenizer")),
                readText(modelNode, "unk_token"));
    }

    private static boolean isGemmaLikeTokenizer(String tokenizerClass, JsonNode tokenizerDocument) {
        if ("GemmaTokenizer".equals(tokenizerClass)) {
            return true;
        }
        JsonNode normalizer = tokenizerDocument.path("normalizer");
        JsonNode preTokenizer = tokenizerDocument.path("pre_tokenizer");
        return "Replace".equals(normalizer.path("type").asText())
                && " ".equals(normalizer.path("pattern").path("String").asText(null))
                && "▁".equals(normalizer.path("content").asText(null))
                && "Split".equals(preTokenizer.path("type").asText())
                && " ".equals(preTokenizer.path("pattern").path("String").asText(null));
    }

    private static boolean isByteLevelBpeTokenizer(String tokenizerClass, JsonNode tokenizerDocument) {
        if ("LlamaTokenizer".equals(tokenizerClass) || "PreTrainedTokenizerFast".equals(tokenizerClass)) {
            return true;
        }
        JsonNode preTokenizer = tokenizerDocument.path("pre_tokenizer");
        JsonNode decoder = tokenizerDocument.path("decoder");
        if ("ByteLevel".equals(preTokenizer.path("type").asText()) || "ByteLevel".equals(decoder.path("type").asText())) {
            return true;
        }
        if ("Sequence".equals(preTokenizer.path("type").asText()) && preTokenizer.path("pretokenizers").isArray()) {
            for (JsonNode child : preTokenizer.path("pretokenizers")) {
                if ("ByteLevel".equals(child.path("type").asText())) {
                    return true;
                }
            }
        }
        if ("Sequence".equals(decoder.path("type").asText()) && decoder.path("decoders").isArray()) {
            for (JsonNode child : decoder.path("decoders")) {
                if ("ByteLevel".equals(child.path("type").asText())) {
                    return true;
                }
            }
        }
        return false;
    }

    private static List<String> readMerges(JsonNode node) {
        if (node == null || node.isNull() || !node.isArray()) {
            return List.of();
        }
        List<String> merges = new ArrayList<>(node.size());
        for (JsonNode child : node) {
            if (child.isTextual()) {
                merges.add(child.asText());
                continue;
            }
            if (child.isArray() && child.size() == 2 && child.get(0).isTextual() && child.get(1).isTextual()) {
                merges.add(child.get(0).asText() + " " + child.get(1).asText());
                continue;
            }
            String maybeToken = readTokenValue(child);
            if (maybeToken != null) {
                merges.add(maybeToken);
            }
        }
        return List.copyOf(merges);
    }

    private static JsonNode readOptionalJson(ObjectMapper objectMapper, File file) throws IOException {
        return file.exists() ? objectMapper.readTree(file) : null;
    }

    private static String readText(JsonNode node, String field) {
        if (node == null) {
            return null;
        }
        JsonNode fieldNode = node.get(field);
        if (fieldNode == null || fieldNode.isNull()) {
            return null;
        }
        return fieldNode.asText();
    }

    private static boolean readBoolean(JsonNode node, String field, boolean defaultValue) {
        if (node == null) {
            return defaultValue;
        }
        JsonNode fieldNode = node.get(field);
        return fieldNode == null || fieldNode.isNull() ? defaultValue : fieldNode.asBoolean();
    }

    private static BigInteger readBigInteger(JsonNode node, String field, BigInteger defaultValue) {
        if (node == null) {
            return defaultValue;
        }
        JsonNode fieldNode = node.get(field);
        if (fieldNode == null || fieldNode.isNull()) {
            return defaultValue;
        }
        return fieldNode.canConvertToLong() ? BigInteger.valueOf(fieldNode.asLong()) : new BigInteger(fieldNode.asText());
    }

    private static String readTokenValue(JsonNode node) {
        if (node == null || node.isNull() || node.isMissingNode()) {
            return null;
        }
        if (node.isTextual()) {
            return node.asText();
        }
        JsonNode contentNode = node.get("content");
        return contentNode == null || contentNode.isNull() ? null : contentNode.asText();
    }

    private static List<String> readStringList(JsonNode node) {
        if (node == null || node.isNull() || !node.isArray()) {
            return List.of();
        }
        List<String> values = new ArrayList<>(node.size());
        for (JsonNode child : node) {
            String value = readTokenValue(child);
            if (value != null) {
                values.add(value);
            }
        }
        return List.copyOf(values);
    }

    private static PreTokenizerConfig parsePreTokenizer(JsonNode preTokenizerNode) {
        if (preTokenizerNode == null || preTokenizerNode.isNull() || preTokenizerNode.isMissingNode()) {
            return new PreTokenizerConfig(null, false, false);
        }

        String splitPattern = null;
        boolean addPrefixSpace = false;
        boolean useRegex = false;

        if ("Sequence".equals(preTokenizerNode.path("type").asText()) && preTokenizerNode.path("pretokenizers").isArray()) {
            for (JsonNode child : preTokenizerNode.path("pretokenizers")) {
                if ("Split".equals(child.path("type").asText())) {
                    splitPattern = child.path("pattern").path("Regex").asText(null);
                }
                if ("ByteLevel".equals(child.path("type").asText())) {
                    addPrefixSpace = readBoolean(child, "add_prefix_space", false);
                    useRegex = readBoolean(child, "use_regex", false);
                }
            }
            return new PreTokenizerConfig(splitPattern, addPrefixSpace, useRegex);
        }

        if ("ByteLevel".equals(preTokenizerNode.path("type").asText())) {
            return new PreTokenizerConfig(
                    null,
                    readBoolean(preTokenizerNode, "add_prefix_space", false),
                    readBoolean(preTokenizerNode, "use_regex", false));
        }

        if ("Split".equals(preTokenizerNode.path("type").asText())) {
            return new PreTokenizerConfig(preTokenizerNode.path("pattern").path("Regex").asText(null), false, false);
        }

        return new PreTokenizerConfig(null, false, false);
    }

    private static PaddingSide parsePaddingSide(String side) {
        if (side == null) {
            return PaddingSide.RIGHT;
        }
        return "left".equalsIgnoreCase(side) ? PaddingSide.LEFT : PaddingSide.RIGHT;
    }

    private static TruncationSide parseTruncationSide(String side) {
        if (side == null) {
            return TruncationSide.RIGHT;
        }
        return "left".equalsIgnoreCase(side) ? TruncationSide.LEFT : TruncationSide.RIGHT;
    }

    public static class OwnerName {
        private final String owner;
        private final String name;

        public OwnerName(String owner, String name) {
            this.owner = owner;
            this.name = name;
        }
    }

    private static class ModelPath {
        private final Path path;

        private ModelPath(Path path) {
            this.path = path;
        }
    }

    public static class OwnerNameOrPath {
        private final OwnerName ownerName;
        private final ModelPath modelPath;

        public OwnerNameOrPath(OwnerName ownerName) {
            this.ownerName = ownerName;
            this.modelPath = null;
        }

        public OwnerNameOrPath(Path modelPath) {
            this.modelPath = new ModelPath(modelPath);
            this.ownerName = null;
        }

        public static OwnerNameOrPath fromPath(Path modelPath) {
            return new OwnerNameOrPath(modelPath);
        }
    }
}
