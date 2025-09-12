package io.teknek.deliverance.tokenizer;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.ImmutableBiMap;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static io.teknek.deliverance.JsonUtils.om;

public class TokenizerModel {

    private static final java.util.regex.Pattern gpt2Pattern = java.util.regex.Pattern.compile(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    );
    @JsonProperty("type")
    public final String type;

    @JsonProperty("unk_token")
    public final String unkToken;

    @JsonProperty("fuse_unk")
    public final boolean fuseUnk;

    @JsonProperty("byte_fallback")
    public final boolean byteFallback;

    @JsonProperty("vocab")
    public final BiMap<String, Long> vocabLookup;

    @JsonProperty("merges")
    public final Map<String, Long> merges;

    private PreTokenizer preTokenizer;
    //private Normalizer normalizer;
    private BiMap<String, Long> addedTokens = HashBiMap.create();
    private BiMap<String, Long> specialTokens = HashBiMap.create();

    private java.util.regex.Pattern addedTokenPattern;

    // This is pretty much a hack to support the legacy tokenizer
    private boolean legacy = false;

    private Optional<Map<String, String>> promptTemplates = Optional.empty();
    private boolean hasToolSupport = false;
    private String eosToken = "";
    private String bosToken = "";
    private final boolean ignoreMerges;

    @JsonCreator
    public TokenizerModel( @JsonProperty("type") String type, @JsonProperty("unk_token") String unkToken,
                           @JsonProperty("fuse_unk") boolean fuseUnk, @JsonProperty("byte_fallback") boolean byteFallback,
            @JsonProperty("vocab") Map<String, Long> vocabLookup, @JsonProperty("ignore_merges") Boolean ignoreMerges,
            @JsonProperty("merges") List<Object> merges) {
        this.type = type;
        this.unkToken = unkToken;
        this.fuseUnk = fuseUnk;
        this.byteFallback = byteFallback;
        this.vocabLookup = HashBiMap.create(vocabLookup);
        this.ignoreMerges = ignoreMerges != null && ignoreMerges;
        this.merges = new HashMap<>();
        if (merges != null) {
            for (int i = 0; i < merges.size(); i++) {
                if (merges.get(i) instanceof String) {
                    this.merges.put((String) merges.get(i), (long) i);
                } else if (merges.get(i) instanceof List) {
                    List<String> merge = (List<String>) merges.get(i);
                    this.merges.put(merge.get(0) + " " + merge.get(1), (long) i);
                } else {
                    throw new IllegalArgumentException("Invalid merge format: " + merges.get(i));
                }
            }
        }
    }

    public static TokenizerModel load(File tokenizerJson){
        JsonNode rootNode = null;
        try {
            rootNode = om.readTree(tokenizerJson);
            if (!rootNode.has("model")) {
                throw new IllegalArgumentException("Json missing 'model' key");
            }
            TokenizerModel model = om.treeToValue(rootNode.get("model"), TokenizerModel.class);
            if (rootNode.has("added_tokens") && rootNode.get("added_tokens") != null) {
                List<Map<String, Object>> addedTokens = om.convertValue(rootNode.get("added_tokens"), List.class);
                model.setAddedTokens(addedTokens);
            }
            if (rootNode.has("pre_tokenizer") && rootNode.get("pre_tokenizer") != null) model.setPreTokenizer(
                    om.treeToValue(rootNode.get("pre_tokenizer"), PreTokenizer.class)
            );
            return model;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private void setPreTokenizer(PreTokenizer preTokenizer) {
        this.preTokenizer = preTokenizer;
    }

    public void setAddedTokens(List<Map<String, Object>> addedTokens) {
        if (addedTokens != null && !addedTokens.isEmpty()) {
            for (Map<String, Object> token : addedTokens) {
                this.addedTokens.put((String) token.get("content"), ((Integer) token.get("id")).longValue());
                this.vocabLookup.put((String) token.get("content"), ((Integer) token.get("id")).longValue());
                if (token.containsKey("special") && (Boolean) token.get("special")) {
                    this.specialTokens.put((String) token.get("content"), ((Integer) token.get("id")).longValue());
                }
            }
            this.addedTokens = ImmutableBiMap.copyOf(this.addedTokens);
            this.specialTokens = ImmutableBiMap.copyOf(this.specialTokens);
            StringBuilder regex = new StringBuilder();
            List<String> delimiters = new ArrayList<>(this.addedTokens.keySet());
            for (int i = 0; i < delimiters.size(); i++) {
                if (i != 0) {
                    regex.append("|");
                }
                regex.append(java.util.regex.Pattern.quote(delimiters.get(i)));
            }
            this.addedTokenPattern = java.util.regex.Pattern.compile(regex.toString());
        }
    }

    public static String[] split(java.util.regex.Pattern p, CharSequence input, int limit, boolean withDelimiters) {
        int matchCount = 0;
        int index = 0;
        boolean matchLimited = limit > 0;
        ArrayList<String> matchList = new ArrayList<>();
        Matcher m = p.matcher(input);

        // Add segments before each match found
        while (m.find()) {
            if (!matchLimited || matchCount < limit - 1) {
                if (index == 0 && index == m.start() && m.start() == m.end()) {
                    // no empty leading substring included for zero-width match
                    // at the beginning of the input char sequence.
                    continue;
                }
                String match = input.subSequence(index, m.start()).toString();
                matchList.add(match);
                index = m.end();
                if (withDelimiters) {
                    matchList.add(input.subSequence(m.start(), index).toString());
                }
                ++matchCount;
            } else if (matchCount == limit - 1) { // last one
                String match = input.subSequence(index, input.length()).toString();
                matchList.add(match);
                index = m.end();
                ++matchCount;
            }
        }

        // If no match was found, return this
        if (index == 0) return new String[] { input.toString() };

        // Add remaining segment
        if (!matchLimited || matchCount < limit) matchList.add(input.subSequence(index, input.length()).toString());

        // Construct result
        int resultSize = matchList.size();
        if (limit == 0) {
            while (resultSize > 0 && matchList.get(resultSize - 1).isEmpty()) {
                resultSize--;
            }
        }
        String[] result = new String[resultSize];
        return matchList.subList(0, resultSize).toArray(result);
    }

    public Pattern getAddedTokenPattern() {
        return addedTokenPattern;
    }
}
