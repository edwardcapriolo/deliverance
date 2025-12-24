package io.teknek.deliverance.grace.bert;

import io.teknek.deliverance.grace.PaddingSide;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.grace.TruncationSide;

import java.math.BigInteger;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/*
        vocab_file: Optional[str] = None,
        do_lower_case: bool = False,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        tokenize_chinese_chars: bool = True,
        strip_accents: Optional[bool] = None,
        vocab: Optional[dict] = None,
 */
public class BertTokenizer extends PreTrainedTokenizer {
    public BertTokenizer(Map<String, String> modelSpecificSpecialTokens, Optional<BigInteger> maxLen,
                         Optional<PaddingSide> paddingSide, Optional<TruncationSide> truncationSide,
                         Optional<Boolean> cleanUpTokenizationSpaces, Optional<Boolean> splitSpecialTokens,
                         Optional<Object> backend, Optional<List<Object>> filesLoaded) {
        super(modelSpecificSpecialTokens, maxLen, paddingSide, truncationSide, cleanUpTokenizationSpaces, splitSpecialTokens, backend, filesLoaded);
    }

    @Override
    public int vocabSize() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Map<String, Integer> getVocab() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int getVocabSize() {
        return 0;
    }
}
