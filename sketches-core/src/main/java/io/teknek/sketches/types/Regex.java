package io.teknek.sketches.types;

public class Regex extends Term {
    private final String pattern;

    public Regex(String pattern) {
        this.pattern = pattern;
    }

    public String pattern() {
        return pattern;
    }
}
