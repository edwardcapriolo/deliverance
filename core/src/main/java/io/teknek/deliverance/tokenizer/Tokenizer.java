package io.teknek.deliverance.tokenizer;

import io.teknek.deliverance.safetensors.prompt.PromptSupport;

import java.util.List;
import java.util.Optional;

public interface Tokenizer {

    List<String> tokenize(String sentence);
    long [] encode(String sentence);
    String decode(long id);
    String decode(long [] ids);
    TokenizerModel getModel();

    String preProcess(String sentence);


    /**
     * Get the prompt support for this tokenizer model if it exists
     * @return prompt support
     */
    Optional<PromptSupport> promptSupport();

    /** quen returs like ĠThe where G is space as the model is trained this way. Other things like 0 or . lack
     * the space. clean up to retun to user*/
    String tokenForResponse(String token);

}
