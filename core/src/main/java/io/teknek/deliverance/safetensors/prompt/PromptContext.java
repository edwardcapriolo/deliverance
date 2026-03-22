package io.teknek.deliverance.safetensors.prompt;

import java.util.List;
import java.util.Optional;

/**
 * Note: This class only holds a string slightly modofied from the output of prompt support. It could be a type of context
 * connecting information from the request till after the response, but for now it is rather useless abstraction
 */
public class PromptContext {
    private final String prompt;
    //private final Optional<List<Tool>> optionalTools;

    public static PromptContext of(String prompt) {
        return new PromptContext(prompt);
    }


    PromptContext(String prompt) {
        this.prompt = prompt;
    }

    public String getPrompt() {
        return prompt;
    }

    @Override
    public String toString() {
        return "PromptContext{" +
                "prompt='" + prompt + '\'' +
                '}';
    }
}

