package io.teknek.sketches;

public class SketchesGenerationException extends RuntimeException {
    public SketchesGenerationException(String message, Throwable cause) {
        super(message, cause);
    }

    public SketchesGenerationException(String message) {
        super(message);
    }
}
