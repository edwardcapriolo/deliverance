package io.teknek.deliverance.grace;

import io.teknek.deliverance.grace.models.TokenizerConfig;

import java.math.BigInteger;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public abstract class PreTrainedTokenizerBase {
    public static final List<String> SPECIAL_TOKEN_ATTRIBUTES = List.of(
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token");

    private final Map<String, String> specialTokensMap;
    private final List<String> additionalSpecialTokens;
    private final SortedMap<Integer, AddedToken> addedTokensDecoder;
    private final Map<String, Integer> addedVocab;
    private final BigInteger modelMaxLength;
    private final PaddingSide paddingSide;
    private final TruncationSide truncationSide;
    private final boolean cleanUpTokenizationSpaces;
    private final boolean splitSpecialTokens;
    private final Object backend;
    private final List<Object> filesLoaded;
    private final int padTokenTypeId;

    PreTrainedTokenizerBase(Map<String, String> modelSpecificSpecialTokens,
                            Optional<BigInteger> maxLen,
                            Optional<PaddingSide> paddingSide,
                            Optional<TruncationSide> truncationSide,
                            Optional<Boolean> cleanUpTokenizationSpaces,
                            Optional<Boolean> splitSpecialTokens,
                            Optional<Object> backend,
                            Optional<List<Object>> filesLoaded,
                            Optional<TokenizerConfig> tokenizerConfig,
                            Optional<SortedMap<Integer, AddedToken>> addedTokensDecoder) {
        TokenizerConfig resolvedConfig = tokenizerConfig.orElse(TokenizerConfig.empty());
        SortedMap<Integer, AddedToken> resolvedAddedTokens = new TreeMap<>(addedTokensDecoder.orElseGet(TreeMap::new));

        this.specialTokensMap = Collections.unmodifiableMap(resolveSpecialTokens(modelSpecificSpecialTokens, resolvedConfig));
        this.additionalSpecialTokens = List.copyOf(resolvedConfig.additionalSpecialTokens());
        this.addedTokensDecoder = Collections.unmodifiableSortedMap(resolvedAddedTokens);
        this.addedVocab = Collections.unmodifiableMap(buildAddedVocab(resolvedAddedTokens));
        this.modelMaxLength = maxLen.orElse(resolvedConfig.modelMaxLength());
        this.paddingSide = paddingSide.orElse(resolvedConfig.paddingSide());
        this.truncationSide = truncationSide.orElse(resolvedConfig.truncationSide());
        this.cleanUpTokenizationSpaces = cleanUpTokenizationSpaces.orElse(resolvedConfig.cleanUpTokenizationSpaces());
        this.splitSpecialTokens = splitSpecialTokens.orElse(resolvedConfig.splitSpecialTokens());
        this.backend = backend.orElse(null);
        this.filesLoaded = List.copyOf(filesLoaded.orElseGet(List::of));
        this.padTokenTypeId = 0;
    }

    public List<String> allSpecialTokens() {
        LinkedHashSet<String> tokens = new LinkedHashSet<>();
        for (String attribute : SPECIAL_TOKEN_ATTRIBUTES) {
            String token = specialTokensMap.get(attribute);
            if (token != null) {
                tokens.add(token);
            }
        }
        tokens.addAll(additionalSpecialTokens);
        return List.copyOf(tokens);
    }

    public List<Integer> allSpecialIds() {
        List<Integer> ids = new ArrayList<>();
        for (String token : allSpecialTokens()) {
            OptionalInt id = tokenToId(token);
            id.ifPresent(ids::add);
        }
        return List.copyOf(ids);
    }

    public abstract Optional<String> chatTemplate();

    public int vocabSize() {
        return getBaseVocab().size();
    }

    public Map<String, Integer> getVocab() {
        LinkedHashMap<String, Integer> combined = new LinkedHashMap<>(getBaseVocab());
        combined.putAll(addedVocab);
        return Collections.unmodifiableMap(combined);
    }

    protected abstract Map<String, Integer> getBaseVocab();

    protected abstract Map<Integer, String> getBaseIdToToken();

    public Map<String, Integer> getAddedVocab() {
        return addedVocab;
    }

    public Map<String, String> specialTokensMap() {
        return specialTokensMap;
    }

    public Optional<String> specialToken(String attribute) {
        return Optional.ofNullable(specialTokensMap.get(attribute));
    }

    public Optional<String> padToken() {
        return specialToken("pad_token");
    }

    public OptionalInt padTokenId() {
        return padToken().isPresent() ? tokenToId(padToken().orElseThrow()) : OptionalInt.empty();
    }

    public Optional<String> bosToken() {
        return specialToken("bos_token");
    }

    public OptionalInt bosTokenId() {
        return bosToken().isPresent() ? tokenToId(bosToken().orElseThrow()) : OptionalInt.empty();
    }

    public Optional<String> eosToken() {
        return specialToken("eos_token");
    }

    public OptionalInt eosTokenId() {
        return eosToken().isPresent() ? tokenToId(eosToken().orElseThrow()) : OptionalInt.empty();
    }

    public Optional<String> unkToken() {
        return specialToken("unk_token");
    }

    public OptionalInt unkTokenId() {
        return unkToken().isPresent() ? tokenToId(unkToken().orElseThrow()) : OptionalInt.empty();
    }

    public BigInteger modelMaxLength() {
        return modelMaxLength;
    }

    public PaddingSide paddingSide() {
        return paddingSide;
    }

    public TruncationSide truncationSide() {
        return truncationSide;
    }

    public boolean cleanUpTokenizationSpaces() {
        return cleanUpTokenizationSpaces;
    }

    public boolean splitSpecialTokens() {
        return splitSpecialTokens;
    }

    public Object backend() {
        return backend;
    }

    public List<Object> filesLoaded() {
        return filesLoaded;
    }

    public int padTokenTypeId() {
        return padTokenTypeId;
    }

    public OptionalInt tokenToId(String token) {
        Integer addedTokenId = addedVocab.get(token);
        if (addedTokenId != null) {
            return OptionalInt.of(addedTokenId);
        }
        Integer baseTokenId = getBaseVocab().get(token);
        return baseTokenId == null ? OptionalInt.empty() : OptionalInt.of(baseTokenId);
    }

    public Optional<String> idToToken(int id) {
        AddedToken addedToken = addedTokensDecoder.get(id);
        if (addedToken != null) {
            return Optional.of(addedToken.content());
        }
        return Optional.ofNullable(getBaseIdToToken().get(id));
    }

    public Tokens tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        for (String segment : splitByAddedTokens(text)) {
            OptionalInt addedTokenId = tokenToId(segment);
            if (addedTokenId.isPresent() && addedVocab.containsKey(segment)) {
                tokens.add(segment);
                continue;
            }
            tokens.addAll(tokenizeSegment(segment));
        }
        return new Tokens(tokens.toArray(new String[0]));
    }

    public Encoding encode(String text) {
        return encode(text, EncodeOptions.defaults());
    }

    public Encoding encode(String text, EncodeOptions options) {
        TokenIds encodedIds = convertTokensToIds(tokenize(text));
        int[] inputIds = encodedIds.asArray();
        if (options.addSpecialTokens()) {
            inputIds = buildInputsWithSpecialTokens(inputIds);
        }

        Encoding encoding = createEncoding(inputIds, inputIds.length);
        if (options.truncation() != null) {
            encoding = truncate(new TokenIds(encoding.inputIds()), options.truncation());
        }
        if (options.padding() != null) {
            encoding = pad(new TokenIds(encoding.inputIds()), options.padding());
        }
        return encoding;
    }

    public BatchEncoding encode(List<String> textBatch, EncodeOptions options) {
        List<TokenIds> encoded = textBatch.stream()
                .map(text -> new TokenIds(encode(text, new EncodeOptions(options.addSpecialTokens(), null, options.truncation())).inputIds()))
                .toList();
        if (options.padding() == null) {
            return new BatchEncoding(encoded.stream().map(ids -> createEncoding(ids.asArray(), ids.length())).toList());
        }
        return pad(encoded, options.padding());
    }

    public TokenIds convertTokensToIds(Tokens tokens) {
        if (tokens.isScalar()) {
            return new TokenIds(resolveTokenId(tokens.getInput()));
        }

        String[] values = tokens.getInputs();
        int[] ids = new int[values.length];
        for (int index = 0; index < values.length; index++) {
            ids[index] = resolveTokenId(values[index]);
        }
        return new TokenIds(ids);
    }

    public Tokens convertIdsToTokens(TokenIds ids, Optional<Boolean> skipSpecialTokens) {
        boolean shouldSkipSpecialTokens = skipSpecialTokens.orElse(false);
        Set<Integer> specialIds = allSpecialIdSet();
        if (ids.isScalar()) {
            if (shouldSkipSpecialTokens && specialIds.contains(ids.getInput())) {
                return new Tokens((String) null);
            }
            return new Tokens(idToToken(ids.getInput()).orElse(null));
        }

        List<String> tokens = new ArrayList<>();
        for (int id : ids.getInputList()) {
            if (shouldSkipSpecialTokens && specialIds.contains(id)) {
                continue;
            }
            idToToken(id).ifPresent(tokens::add);
        }
        return new Tokens(tokens.toArray(new String[0]));
    }

    public Encoding truncate(TokenIds tokenIds, TruncationOptions options) {
        int[] values = tokenIds.asArray();
        if (values.length <= options.maxLength()) {
            return createEncoding(values, values.length);
        }

        int[] truncated;
        if (options.side() == TruncationSide.LEFT) {
            truncated = Arrays.copyOfRange(values, values.length - options.maxLength(), values.length);
        } else {
            truncated = Arrays.copyOf(values, options.maxLength());
        }
        return createEncoding(truncated, truncated.length);
    }

    public Encoding pad(TokenIds tokenIds, PaddingOptions options) {
        return pad(List.of(tokenIds), options).encodings().getFirst();
    }

    public BatchEncoding pad(List<TokenIds> batch, PaddingOptions options) {
        if (batch.isEmpty()) {
            return new BatchEncoding(List.of());
        }

        if (options.strategy() == PaddingStrategy.DO_NOT_PAD && options.padToMultipleOf() == null) {
            List<Encoding> encodings = batch.stream()
                    .map(tokenIds -> createEncoding(tokenIds.asArray(), tokenIds.length()))
                    .toList();
            return new BatchEncoding(encodings);
        }

        int targetLength = resolvePaddingLength(batch, options);
        if (targetLength < 0) {
            throw new IllegalArgumentException("Padding target length must be non-negative");
        }

        if (options.padToMultipleOf() != null && targetLength > 0) {
            targetLength = roundUpToMultiple(targetLength, options.padToMultipleOf());
        }

        int padTokenId = padTokenId().orElseThrow(() -> new IllegalStateException("Tokenizer has no pad token configured"));
        PaddingSide effectiveSide = options.side() == null ? paddingSide : options.side();
        List<Encoding> encodings = new ArrayList<>(batch.size());
        for (TokenIds tokenIds : batch) {
            int[] values = tokenIds.asArray();
            if (values.length >= targetLength) {
                encodings.add(createEncoding(values, values.length));
                continue;
            }

            int[] padded = new int[targetLength];
            int[] attentionMask = new int[targetLength];
            if (effectiveSide == PaddingSide.LEFT) {
                int padWidth = targetLength - values.length;
                Arrays.fill(padded, 0, padWidth, padTokenId);
                System.arraycopy(values, 0, padded, padWidth, values.length);
                Arrays.fill(attentionMask, padWidth, targetLength, 1);
            } else {
                System.arraycopy(values, 0, padded, 0, values.length);
                Arrays.fill(padded, values.length, targetLength, padTokenId);
                Arrays.fill(attentionMask, 0, values.length, 1);
            }
            encodings.add(new Encoding(padded, attentionMask, computeSpecialTokensMask(padded)));
        }
        return new BatchEncoding(encodings);
    }

    protected Set<Integer> allSpecialIdSet() {
        return Set.copyOf(allSpecialIds());
    }

    protected SortedMap<Integer, AddedToken> addedTokensDecoder() {
        return addedTokensDecoder;
    }

    protected Optional<BytePairEncodingModel> bytePairEncodingModel() {
        return Optional.empty();
    }

    protected int[] buildInputsWithSpecialTokens(int[] inputIds) {
        return inputIds;
    }

    private int resolveTokenId(String token) {
        OptionalInt tokenId = tokenToId(token);
        if (tokenId.isPresent()) {
            return tokenId.getAsInt();
        }
        OptionalInt unknownId = unkTokenId();
        return unknownId.orElse(-1);
    }

    private LinkedHashMap<String, String> resolveSpecialTokens(Map<String, String> modelSpecificSpecialTokens,
                                                               TokenizerConfig tokenizerConfig) {
        LinkedHashMap<String, String> resolved = new LinkedHashMap<>();
        for (String attribute : SPECIAL_TOKEN_ATTRIBUTES) {
            String value = tokenizerConfig.specialTokensMap().get(attribute);
            if (value == null) {
                value = modelSpecificSpecialTokens.get(attribute);
            }
            if (value != null) {
                resolved.put(attribute, value);
            }
        }
        for (Map.Entry<String, String> entry : modelSpecificSpecialTokens.entrySet()) {
            resolved.putIfAbsent(entry.getKey(), entry.getValue());
        }
        return resolved;
    }

    private Map<String, Integer> buildAddedVocab(SortedMap<Integer, AddedToken> decoder) {
        Map<String, Integer> vocab = new LinkedHashMap<>(decoder.size());
        for (Map.Entry<Integer, AddedToken> entry : decoder.entrySet()) {
            vocab.put(entry.getValue().content(), entry.getKey());
        }
        return vocab;
    }

    protected Encoding createEncoding(int[] inputIds, int tokenCount) {
        int[] attentionMask = new int[inputIds.length];
        Arrays.fill(attentionMask, 0, tokenCount, 1);
        return new Encoding(inputIds, attentionMask, computeSpecialTokensMask(inputIds));
    }

    private int[] computeSpecialTokensMask(int[] ids) {
        Set<Integer> specialIds = allSpecialIdSet();
        int[] mask = new int[ids.length];
        for (int index = 0; index < ids.length; index++) {
            mask[index] = specialIds.contains(ids[index]) ? 1 : 0;
        }
        return mask;
    }

    private int resolvePaddingLength(List<TokenIds> batch, PaddingOptions options) {
        return switch (options.strategy()) {
            case DO_NOT_PAD -> options.padToMultipleOf() == null
                    ? -1
                    : batch.stream().mapToInt(TokenIds::length).max().orElse(0);
            case LONGEST -> batch.stream().mapToInt(TokenIds::length).max().orElse(0);
            case MAX_LENGTH -> options.maxLength() == null ? 0 : options.maxLength();
        };
    }

    private int roundUpToMultiple(int value, int multiple) {
        if (value == 0) {
            return 0;
        }
        int remainder = value % multiple;
        return remainder == 0 ? value : value + multiple - remainder;
    }

    protected List<String> splitByAddedTokens(String text) {
        if (splitSpecialTokens || addedVocab.isEmpty() || text.isEmpty()) {
            return List.of(text);
        }

        List<AddedToken> addedTokens = addedTokensDecoder.values().stream()
                .sorted(Comparator.comparingInt((AddedToken token) -> token.content().length()).reversed())
                .toList();
        List<String> segments = new ArrayList<>();
        StringBuilder plainText = new StringBuilder();
        int index = 0;
        while (index < text.length()) {
            AddedToken matchedToken = null;
            for (AddedToken candidate : addedTokens) {
                if (matchesAddedToken(text, index, candidate)) {
                    matchedToken = candidate;
                    break;
                }
            }

            if (matchedToken == null) {
                plainText.append(text.charAt(index));
                index++;
                continue;
            }

            if (matchedToken.lstrip()) {
                stripTrailingWhitespace(plainText);
            }
            if (plainText.length() > 0) {
                segments.add(plainText.toString());
                plainText.setLength(0);
            }
            segments.add(matchedToken.content());
            index += matchedToken.content().length();
            if (matchedToken.rstrip()) {
                while (index < text.length() && Character.isWhitespace(text.charAt(index))) {
                    index++;
                }
            }
        }

        if (plainText.length() > 0) {
            segments.add(plainText.toString());
        }
        return segments;
    }

    private boolean matchesAddedToken(String text, int index, AddedToken candidate) {
        if (!text.startsWith(candidate.content(), index)) {
            return false;
        }
        if (!candidate.singleWord()) {
            return true;
        }

        int start = index;
        int end = index + candidate.content().length();
        boolean leftBoundary = start == 0 || isBoundary(text.codePointBefore(start));
        boolean rightBoundary = end == text.length() || isBoundary(text.codePointAt(end));
        return leftBoundary && rightBoundary;
    }

    private boolean isBoundary(int codePoint) {
        return PreTrainedTokenizer.isControl(codePoint)
                || PreTrainedTokenizer.isPunctuation(codePoint)
                || PreTrainedTokenizer.isWhitespace(codePoint);
    }

    private void stripTrailingWhitespace(StringBuilder plainText) {
        while (plainText.length() > 0 && Character.isWhitespace(plainText.charAt(plainText.length() - 1))) {
            plainText.setLength(plainText.length() - 1);
        }
    }

    protected List<String> tokenizeSegment(String text) {
        Optional<BytePairEncodingModel> optionalModel = bytePairEncodingModel();
        if (optionalModel.isEmpty()) {
            throw new UnsupportedOperationException("Tokenizer does not have an encode model");
        }
        BytePairEncodingModel model = optionalModel.orElseThrow();
        String normalized = normalizeForBpeInput(text, model);
        List<String> pretokenized = pretokenize(normalized, model.preTokenizer());
        List<String> tokens = new ArrayList<>();
        for (String piece : pretokenized) {
            if (piece.isEmpty()) {
                continue;
            }
            tokens.addAll(applyBpe(piece, model));
        }
        return tokens;
    }

    protected String normalizeForBpeInput(String text, BytePairEncodingModel model) {
        return text;
    }

    protected List<String> pretokenize(String text, PreTokenizerConfig config) {
        if (text.isEmpty()) {
            return List.of();
        }

        String normalized = config.addPrefixSpace() && !text.startsWith(" ") ? " " + text : text;
        String splitPattern = config.effectiveSplitPattern();
        if (splitPattern == null) {
            return List.of(normalized);
        }

        Pattern pattern = Pattern.compile(splitPattern);
        Matcher matcher = pattern.matcher(normalized);
        List<String> pieces = new ArrayList<>();
        int cursor = 0;
        while (matcher.find()) {
            if (matcher.start() > cursor) {
                pieces.add(normalized.substring(cursor, matcher.start()));
            }
            pieces.add(matcher.group());
            cursor = matcher.end();
        }
        if (cursor < normalized.length()) {
            pieces.add(normalized.substring(cursor));
        }
        if (pieces.isEmpty()) {
            pieces.add(normalized);
        }
        return pieces;
    }

    protected String encodeForBpe(String token, BytePairEncodingModel model) {
        return ByteLevelCodec.encode(token);
    }

    protected List<String> applyBpe(String token, BytePairEncodingModel model) {
        String encoded = encodeForBpe(token, model);
        if (model.vocab().containsKey(encoded)) {
            return List.of(encoded);
        }

        List<String> symbols = encoded.codePoints()
                .mapToObj(codePoint -> new String(Character.toChars(codePoint)))
                .collect(java.util.stream.Collectors.toCollection(ArrayList::new));

        while (symbols.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            int bestIndex = -1;
            for (int index = 0; index < symbols.size() - 1; index++) {
                String pair = symbols.get(index) + " " + symbols.get(index + 1);
                Integer rank = model.mergeRanks().get(pair);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIndex = index;
                }
            }

            if (bestIndex < 0) {
                break;
            }

            List<String> merged = new ArrayList<>(symbols.size() - 1);
            for (int index = 0; index < symbols.size(); index++) {
                if (index == bestIndex) {
                    merged.add(symbols.get(index) + symbols.get(index + 1));
                    index++;
                } else {
                    merged.add(symbols.get(index));
                }
            }
            symbols = merged;
        }

        List<String> output = new ArrayList<>(symbols.size());
        for (String symbol : symbols) {
            if (model.vocab().containsKey(symbol)) {
                output.add(symbol);
            } else if (model.unkToken() != null) {
                output.add(model.unkToken());
            } else {
                output.add(symbol);
            }
        }
        return output;
    }

    public static PreTrainedTokenizer from_pretrained(Class<PreTrainedTokenizer> clazz) {
        return null;
    }
}
