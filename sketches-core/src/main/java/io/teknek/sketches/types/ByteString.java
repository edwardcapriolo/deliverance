package io.teknek.sketches.types;

public class ByteString extends Term {
    private final String value;

    public ByteString(String value) {
        this.value = value;
    }

    public String value() {
        return value;
    }
}
