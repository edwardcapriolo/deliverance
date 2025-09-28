package io.teknek.deliverance.tokenizer;

import io.teknek.deliverance.safetensors.prompt.PromptSupport;

import java.util.List;
import java.util.Optional;

public interface Tokenizer {

    List<String> tokenize(String sentence);
    long [] encode(String sentence);
    String decode(long id);
    TokenizerModel getModel();

    String preProcess(String sentence);


    /**
     * Get the prompt support for this tokenizer model if it exists
     * @return prompt support
     */
    Optional<PromptSupport> promptSupport();


}
