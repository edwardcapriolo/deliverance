package io.teknek.deliverance.grace;

import io.teknek.dysfx.multiple.Tuple2;

import java.math.BigInteger;
import java.util.*;

public abstract class PreTrainedTokenizer extends PreTrainedTokenizerBase{
    PreTrainedTokenizer(Map<String, String> modelSpecificSpecialTokens, Optional<BigInteger> maxLen, Optional<PaddingSide> paddingSide, Optional<TruncationSide> truncationSide, Optional<Boolean> cleanUpTokenizationSpaces, Optional<Boolean> splitSpecialTokens, Optional<Object> backend, Optional<List<Object>> filesLoaded) {
        super(modelSpecificSpecialTokens, maxLen, paddingSide, truncationSide, cleanUpTokenizationSpaces, splitSpecialTokens, backend, filesLoaded);
    }

    public static boolean isWhitespace(int c){
        //or char == "\t" or char == "\n" or char == "\r":
        if ( c == ' ' || c == '\n' || c == '\t' || c == '\r'){
            return true;
        }
        return Character.isWhitespace(c);
    }

    public static boolean isControl(byte b){
        return isControl((int)b);
    }

    public static boolean isControl(int c){
        if (c == '\t' || c == '\n' || c == '\r') {
            return false;
        }
        return Character.isISOControl(c);
    }

    public static boolean isEndOfWord(String str){
        int lastCodepointStartIndex = str.offsetByCodePoints(str.length(), -1);
        int last = str.codePointAt(lastCodepointStartIndex);
        return isControl(last) | isPunctuation(last)| isWhitespace(last);
    }

    public static boolean isStartOfWord(String str){
        OptionalInt first = str.codePoints().findFirst();
        if (first.isPresent()){
            int firstInt = first.getAsInt();
            return isControl(firstInt) | isPunctuation(firstInt)| isWhitespace(firstInt);
        }
        return false;
    }


    public static boolean isPunctuation(int cp){
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64)
                || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)){
            return true;
        }
        int type = Character.getType(cp);
        return type == Character.DASH_PUNCTUATION || type == Character.START_PUNCTUATION ||
                type == Character.END_PUNCTUATION || type == Character.CONNECTOR_PUNCTUATION ||
                type == Character.OTHER_PUNCTUATION || type == Character.INITIAL_QUOTE_PUNCTUATION ||
                type == Character.FINAL_QUOTE_PUNCTUATION;
    }

    public abstract int getVocabSize();


    public String decode(TokenIds tokenIds, boolean skipSpecialTokens,
                  boolean cleanUpTokenizationSpaces,
                  boolean spacesBetweenSpecialTokens,
                  boolean useSourceTokenizer){
        //# If given is a single id, prevents splitting the string in upcoming loop
        ///filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        throw new UnsupportedOperationException();
    }

    //str or list[str]
    public Tokens convertIdsToTokens(TokenIds tokenIds, boolean skipSpecialTokens){
        if( tokenIds.isScalar()){
            return new Tokens("bla");
        }
        List<String> tokens = new ArrayList<>();
        for (int i = 0 ;i<tokenIds.getInputList().length; i++){
            int idx = tokenIds.getInputList()[i];
            if (skipSpecialTokens && (allSpecialIds().contains(idx))){
                continue;
            }
            /*
             tokens.append(
                self._added_tokens_decoder[index].content
                if index in self._added_tokens_decoder
                else self._convert_id_to_token(index)
            )
             */
        }
        return new Tokens(tokens.toArray(new String[0]));
    }
/*
TextInput = str
PreTokenizedInput = list[str]
EncodedInput = list[int]
TextInputPair = tuple[str, str]
PreTokenizedInputPair = tuple[list[str], list[str]]
EncodedInputPair = tuple[list[int], list[int]]
 */
    static class PreTokenizedInput extends ArrayList<String> {

    }
    static class EncodedInput extends ArrayList<Integer> {

    }
    static class TextInputPair extends Tuple2<String, String> {
        public TextInputPair(String s, String s2) {
            super(s, s2);
        }
    }
    static class EncodedInputPair extends io.teknek.dysfx.multiple.Tuple2<List<Integer>, List<Integer>>{

        public EncodedInputPair(List<Integer> integers, List<Integer> integers2) {
            super(integers, integers2);
        }
    }
    static class PreTokenizedInputPair extends io.teknek.dysfx.multiple.Tuple2<List<String>,List<String>> {
        public PreTokenizedInputPair(List<String> strings, List<String> strings2) {
            super(strings, strings2);
        }
    }
}
