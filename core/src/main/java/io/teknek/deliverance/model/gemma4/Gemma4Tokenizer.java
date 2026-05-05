package io.teknek.deliverance.model.gemma4;

import io.teknek.deliverance.model.gemma3.Gemma3Tokenizer;

import java.nio.file.Path;

public class Gemma4Tokenizer extends Gemma3Tokenizer {
    public Gemma4Tokenizer(Path modelRoot) {
        super(modelRoot);
    }
}
