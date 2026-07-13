package io.teknek.sketches.types;

public class JsonSchema extends Term {
    private final String schema;

    public JsonSchema(String schema) {
        this.schema = schema;
    }

    public String schema() {
        return schema;
    }
}
