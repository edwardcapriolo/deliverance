package io.teknek.deliverance.safetensors.prompt;



import java.util.List;
import java.util.Optional;

public class PromptContext {
    private final String prompt;
    private final Optional<List<Tool>> optionalTools;

    public static PromptContext of(String prompt) {
        return new PromptContext(prompt);
    }

    PromptContext(String prompt, Optional<List<Tool>> optionalTools) {
        this.prompt = prompt;
        this.optionalTools = optionalTools;
    }

    PromptContext(String prompt) {
        this.prompt = prompt;
        this.optionalTools = Optional.empty();
    }

    public boolean hasTools() {
        return optionalTools.isPresent();
    }

    public Optional<List<Tool>> getTools() {
        return optionalTools;
    }

    public String getPrompt() {
        return prompt;
    }

    @Override
    public String toString() {
        return "PromptContext{" + "prompt='" + prompt + '\'' + ", optionalTools=" + optionalTools + '}';
    }
}

