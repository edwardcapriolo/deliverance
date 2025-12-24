package io.teknek.deliverance.grace;

import java.math.BigInteger;
import java.util.*;

public abstract class PreTrainedTokenizerBase {
    public final List<String> SPECIAL_TOKEN_ATTRIBUTES = List.of( "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token");
    private Map<String,String> vocabFileNames;
    private Map<String,Map<String,String>> pretrainedVocabFilesMap = new HashMap<>();
    private Optional<String> autoClass;
    private int padTokenTypeId;
    private Map<String, AddedToken> specialTokensMap ;
    private List<Object> extraSpecialTokens;
    private BigInteger modelMaxLength;
    private boolean cleanUpTokenizationSpaces;
    private boolean splitSpecialTokens;
    private Object backend;
    private List<Object> filesLoaded;

    PreTrainedTokenizerBase(Map<String,String> modelSpecificSpecialTokens,
                            Optional<BigInteger> maxLen, Optional<PaddingSide> paddingSide,
                            Optional<TruncationSide> truncationSide,
                            Optional<Boolean> cleanUpTokenizationSpaces,
                            Optional<Boolean> splitSpecialTokens,
                            Optional<Object> backend,
                            Optional<List<Object>> filesLoaded){
        padTokenTypeId = 0;
        specialTokensMap = (Map<String, AddedToken>) fromKeys(SPECIAL_TOKEN_ATTRIBUTES);
        extraSpecialTokens = new ArrayList<>();
        Map<String,String> autoModelSpecificTokens= new HashMap<>();
        this.modelMaxLength = maxLen.isEmpty() ? TokenizerUtils.VERY_LARGE_INTEGER : maxLen.get();
        cleanUpTokenizationSpaces.ifPresent( value -> this.cleanUpTokenizationSpaces = value);
        splitSpecialTokens.ifPresent(value -> this.splitSpecialTokens = value);
        this.backend = backend.orElse(null);
        this.filesLoaded = filesLoaded.orElse(new ArrayList<>());
    }

    //def all_special_tokens(self) -> list[str]:
    public List<String> allSpecialTokens(){
        throw new UnsupportedOperationException();
    }

    public List<Integer> allSpecialIds(){
        throw new UnsupportedOperationException();
    }
    public  static <T> Map<T, ?> fromKeys(List<T> keys){
        Map<T, ?> newMap = new HashMap<>(keys.size());
        keys.forEach(key -> newMap.put(key, null));
        return newMap;
    }

    public abstract int vocabSize();

    public abstract Map<String, Integer> getVocab();

    public TokenIds convertTokensToIds(Tokens tokens){
        throw new UnsupportedOperationException("not yet");
        //return [self._convert_token_to_id_with_added_voc(token) for token in tokens]
    }

    /**
     *
     * @param ids
     * @param skipSpecialTokens Whether or not to remove special tokens in the decoding.
     */
    public Tokens convertIdsToTokens(TokenIds ids, Optional<Boolean> skipSpecialTokens){
        throw new UnsupportedOperationException("not yet");
    }


    public static PreTrainedTokenizer from_pretrained(Class<PreTrainedTokenizer> clazz){
        return null;
        //clazz.getConstructor()
    }
        /*  def _from_pretrained(
        cls,
        resolved_vocab_files,
        pretrained_model_name_or_path,
        init_configuration,
        *init_inputs,
        token=None,
        cache_dir=None,
        local_files_only=False,
        _commit_hash=None,
        _is_local=False,
        trust_remote_code=False,
        **kwargs,
    ): */
}
