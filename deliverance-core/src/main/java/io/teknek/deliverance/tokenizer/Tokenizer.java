package io.teknek.deliverance.tokenizer;

import java.util.List;

public interface Tokenizer {

    List<String> tokenize(String sentence);
    long [] encode(String sentence);
    String decode(long id);
    TokenizerModel getModel();

    String preProcess(String sentence);
}
