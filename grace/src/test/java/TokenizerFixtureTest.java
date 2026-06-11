import io.teknek.deliverance.grace.*;
import org.junit.jupiter.api.Test;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

public class TokenizerFixtureTest {

    @Test
    void loadsTokenizerFromLocalPathFixture() throws Exception {
        PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(fixturePath());

        assertEquals(18, tokenizer.getVocabSize());
        assertEquals(22, tokenizer.getVocab().size());
        assertEquals(4, tokenizer.getAddedVocab().size());
        assertEquals(32, tokenizer.modelMaxLength().intValueExact());
        assertEquals(PaddingSide.LEFT, tokenizer.paddingSide());
        assertEquals(TruncationSide.RIGHT, tokenizer.truncationSide());
        assertTrue(tokenizer.cleanUpTokenizationSpaces());
        assertFalse(tokenizer.splitSpecialTokens());
    }

    @Test
    void exposesSpecialTokensExactlyLikeTokenizerMetadata() throws Exception {
        PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(fixturePath());

        Map<String, String> expectedSpecialTokenMap = new LinkedHashMap<>();
        expectedSpecialTokenMap.put("eos_token", "<eos>");
        expectedSpecialTokenMap.put("pad_token", "<pad>");
        assertEquals(expectedSpecialTokenMap, tokenizer.specialTokensMap());
        assertEquals(List.of("<eos>", "<pad>", "<assistant>"), tokenizer.allSpecialTokens());
        assertEquals(List.of(101, 100, 102), tokenizer.allSpecialIds());
        assertEquals(Optional.of("<pad>"), tokenizer.padToken());
        assertEquals(OptionalInt.of(100), tokenizer.padTokenId());
        assertEquals(Optional.of("<eos>"), tokenizer.eosToken());
        assertEquals(OptionalInt.of(101), tokenizer.eosTokenId());
        assertTrue(tokenizer.chatTemplate().orElseThrow().startsWith("{%- for message in messages %}"));
    }

    @Test
    void resolvesIdsAndTokensWithAddedVocabulary() throws Exception {
        PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(fixturePath());

        assertEquals(OptionalInt.of(11), tokenizer.tokenToId("Hello"));
        assertEquals(OptionalInt.of(102), tokenizer.tokenToId("<assistant>"));
        assertEquals(OptionalInt.of(103), tokenizer.tokenToId("<tool_call>"));
        assertEquals(OptionalInt.empty(), tokenizer.tokenToId("missing-token"));

        assertEquals(Optional.of("Hello"), tokenizer.idToToken(11));
        assertEquals(Optional.of("<assistant>"), tokenizer.idToToken(102));
        assertEquals(Optional.empty(), tokenizer.idToToken(9999));

        TokenIds ids = tokenizer.convertTokensToIds(new Tokens(new String[]{"Hello", "<assistant>", "<tool_call>"}));
        assertArrayEquals(new int[]{11, 102, 103}, ids.getInputList());

        Tokens tokens = tokenizer.convertIdsToTokens(new TokenIds(new int[]{11, 102, 103}), Optional.of(false));
        assertArrayEquals(new String[]{"Hello", "<assistant>", "<tool_call>"}, tokens.getInputs());

        Tokens withoutSpecialTokens = tokenizer.convertIdsToTokens(new TokenIds(new int[]{11, 102, 17, 101}), Optional.of(true));
        assertArrayEquals(new String[]{"Hello", "Ġworld"}, withoutSpecialTokens.getInputs());
    }

    @Test
    void tokenizesRawTextUsingByteLevelBpe() throws Exception {
        PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(fixturePath());

        assertArrayEquals(new String[]{"Hello", "Ġworld", "!"}, tokenizer.tokenize("Hello world!").getInputs());
        assertArrayEquals(new String[]{"Hello", "Ġ", "Ġworld", "!"}, tokenizer.tokenize("Hello  world!").getInputs());
        assertArrayEquals(new String[]{"Hello", "<assistant>", "Ġworld", "<tool_call>", "!"},
                tokenizer.tokenize("Hello<assistant> world<tool_call>!").getInputs());
    }

    @Test
    void encodesRawTextIntoIdsAndMasks() throws Exception {
        PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(fixturePath());

        Encoding encoding = tokenizer.encode("Hello world!");
        assertArrayEquals(new int[]{11, 17, 4}, encoding.inputIds());
        assertArrayEquals(new int[]{1, 1, 1}, encoding.attentionMask());
        assertArrayEquals(new int[]{0, 0, 0}, encoding.specialTokensMask());

        Encoding withAddedTokens = tokenizer.encode("Hello<assistant> world<tool_call>!");
        assertArrayEquals(new int[]{11, 102, 17, 103, 4}, withAddedTokens.inputIds());
        assertArrayEquals(new int[]{1, 1, 1, 1, 1}, withAddedTokens.attentionMask());
        assertArrayEquals(new int[]{0, 1, 0, 0, 0}, withAddedTokens.specialTokensMask());

        Encoding withRepeatedWhitespace = tokenizer.encode("Hello  world!");
        assertArrayEquals(new int[]{11, 12, 17, 4}, withRepeatedWhitespace.inputIds());
        assertArrayEquals(new int[]{1, 1, 1, 1}, withRepeatedWhitespace.attentionMask());
        assertArrayEquals(new int[]{0, 0, 0, 0}, withRepeatedWhitespace.specialTokensMask());
    }

    @Test
    void encodesBatchesWithTypedPaddingAndTruncationOptions() throws Exception {
        PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(fixturePath());

        BatchEncoding batchEncoding = tokenizer.encode(
                List.of("Hello world!", "Hello"),
                EncodeOptions.defaults().withPadding(PaddingOptions.maxLength(4)).withTruncation(TruncationOptions.maxLength(4)));
        assertEquals(2, batchEncoding.encodings().size());
        assertArrayEquals(new int[]{100, 11, 17, 4}, batchEncoding.encodings().get(0).inputIds());
        assertArrayEquals(new int[]{100, 100, 100, 11}, batchEncoding.encodings().get(1).inputIds());
        assertArrayEquals(new int[]{0, 1, 1, 1}, batchEncoding.encodings().get(0).attentionMask());
        assertArrayEquals(new int[]{0, 0, 0, 1}, batchEncoding.encodings().get(1).attentionMask());
    }

    @Test
    void decodesByteLevelTokensAndOptionallySkipsSpecialTokens() throws Exception {
        PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(fixturePath());

        assertEquals("Hello world!", tokenizer.decode(new TokenIds(new int[]{11, 17, 4}), false, false, false, false));
        assertEquals("Hello world!", tokenizer.decode(new TokenIds(new int[]{11, 100, 17, 101, 4}), true, false, false, false));
        assertEquals("<pad> <eos>", tokenizer.decode(new TokenIds(new int[]{100, 101}), false, false, true, false));
        assertEquals("Hello!", tokenizer.decode(new TokenIds(new int[]{11, 12, 4}), false, true, false, false));
    }

    @Test
    void appliesTypedPaddingOptionsForSingleSequencesAndBatches() throws Exception {
        PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(fixturePath());

        Encoding padded = tokenizer.pad(new TokenIds(new int[]{11, 17}), PaddingOptions.maxLength(4));
        assertArrayEquals(new int[]{100, 100, 11, 17}, padded.inputIds());
        assertArrayEquals(new int[]{0, 0, 1, 1}, padded.attentionMask());
        assertArrayEquals(new int[]{1, 1, 0, 0}, padded.specialTokensMask());

        BatchEncoding batch = tokenizer.pad(
                List.of(new TokenIds(new int[]{11, 17}), new TokenIds(new int[]{11})),
                new PaddingOptions(PaddingStrategy.LONGEST, null, 4, PaddingSide.RIGHT));
        assertEquals(2, batch.encodings().size());
        assertArrayEquals(new int[]{11, 17, 100, 100}, batch.encodings().get(0).inputIds());
        assertArrayEquals(new int[]{11, 100, 100, 100}, batch.encodings().get(1).inputIds());
        assertArrayEquals(new int[]{1, 1, 0, 0}, batch.encodings().get(0).attentionMask());
        assertArrayEquals(new int[]{1, 0, 0, 0}, batch.encodings().get(1).attentionMask());
    }

    @Test
    void truncatesFromEitherSideWithTypedOptions() throws Exception {
        PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(fixturePath());

        Encoding rightTruncated = tokenizer.truncate(new TokenIds(new int[]{11, 17, 4, 101}), TruncationOptions.maxLength(2));
        assertArrayEquals(new int[]{11, 17}, rightTruncated.inputIds());
        assertArrayEquals(new int[]{1, 1}, rightTruncated.attentionMask());
        assertArrayEquals(new int[]{0, 0}, rightTruncated.specialTokensMask());

        Encoding leftTruncated = tokenizer.truncate(
                new TokenIds(new int[]{11, 17, 4, 101}),
                TruncationOptions.maxLength(2).withSide(TruncationSide.LEFT));
        assertArrayEquals(new int[]{4, 101}, leftTruncated.inputIds());
        assertArrayEquals(new int[]{1, 1}, leftTruncated.attentionMask());
        assertArrayEquals(new int[]{0, 1}, leftTruncated.specialTokensMask());
    }

    @Test
    void pretokenizePreservesUnmatchedSegmentsAroundRegexMatches() throws Exception {
        PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(fixturePath());

        assertArrayEquals(new String[]{"Hello", "$", "Ġworld"}, tokenizer.tokenize("Hello$ world").getInputs());
    }

    @Test
    void gemmaFixtureKeepsTurnTokensAtomicAndDecodesSpaces() throws Exception {
        PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(gemmaFixturePath());

        String prompt = """
                <|turn>system
                """;

        Encoding encoding = tokenizer.encode(prompt);
        assertEquals(105, encoding.inputIds()[0]);
        assertEquals(9730, encoding.inputIds()[encoding.inputIds().length - 1]);
        assertEquals(prompt, tokenizer.decode(new TokenIds(encoding.inputIds()), false, false, false, false));

        assertEquals("You are a concise assistant.",
                tokenizer.decode(new TokenIds(new int[]{3048, 659, 496, 63510, 16326, 236761}), false, false, false, false));
    }

    @Test
    void llamaByteFallbackTokensDecodeToUtf8Characters() {
        assertEquals("\n\n# Display", LlamaTokenizer.decodeByteFallback("<0x0A><0x0A># Display"));
        assertEquals("é", LlamaTokenizer.decodeByteFallback("<0xC3><0xA9>"));
    }

    private Path fixturePath() throws URISyntaxException {
        return Path.of(getClass().getResource("/tokenizers/tiny-qwen").toURI());
    }

    private Path gemmaFixturePath() throws URISyntaxException {
        return Path.of(getClass().getResource("/tokenizers/tiny-gemma").toURI());
    }
}
