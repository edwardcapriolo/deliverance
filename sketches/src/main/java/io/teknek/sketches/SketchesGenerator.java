package io.teknek.sketches;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.module.jsonSchema.factories.SchemaFactoryWrapper;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.CausalLanguageModel;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.prompt.PromptContext;

import java.util.List;
import java.util.UUID;

public final class SketchesGenerator {
    private static final ObjectMapper JSON = new ObjectMapper();

    private final CausalLanguageModel model;
    private final GeneratorParameters parameters = new GeneratorParameters();
    private Mode mode = Mode.NONE;
    private String constraint;

    SketchesGenerator(CausalLanguageModel model) {
        this.model = java.util.Objects.requireNonNull(model, "model");
    }

    public SketchesGenerator regex(String regex) {
        mode = Mode.REGEX;
        constraint = java.util.Objects.requireNonNull(regex, "regex");
        return this;
    }

    public SketchesGenerator choice(List<String> choices) {
        mode = Mode.CHOICE;
        parameters.withGuidedChoice(List.copyOf(choices));
        return this;
    }

    public <T> TypedJsonGenerator<T> json(Class<T> type) {
        return json(JSON.constructType(type));
    }

    public <T> TypedJsonGenerator<T> json(TypeReference<T> type) {
        return json(JSON.constructType(type.getType()));
    }

    public String raw(String prompt) {
        return generate(prompt);
    }

    public String generate(String prompt) {
        applyConstraint();
        try {
            Response response = model.generate(
                    UUID.randomUUID(),
                    promptContext(prompt),
                    parameters,
                    new DoNothingGenerateEvent());
            return response.responseText;
        } catch (SketchesGenerationException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new SketchesGenerationException("Guided generation failed", e);
        }
    }

    private void applyConstraint() {
        if (mode == Mode.REGEX) {
            parameters.withGuidedRegex(constraint);
        }
    }

    private PromptContext promptContext(String prompt) {
        if (model.promptSupport().isPresent()) {
            return model.promptSupport().orElseThrow().builder()
                    .addUserMessage(prompt)
                    .build();
        }
        return PromptContext.of(prompt);
    }

    private <T> TypedJsonGenerator<T> json(JavaType type) {
        try {
            SchemaFactoryWrapper visitor = new SchemaFactoryWrapper();
            JSON.acceptJsonFormatVisitor(type, visitor);
            String schema = JSON.writeValueAsString(visitor.finalSchema());
            return new TypedJsonGenerator<>(this, type, schema);
        } catch (Exception e) {
            throw new SketchesGenerationException("Unable to create JSON schema for " + type, e);
        }
    }

    public static final class TypedJsonGenerator<T> {
        private final SketchesGenerator generator;
        private final JavaType type;
        private final String schema;

        private TypedJsonGenerator(SketchesGenerator generator, JavaType type, String schema) {
            this.generator = generator;
            this.type = type;
            this.schema = schema;
        }

        public T generate(String prompt) {
            generator.mode = Mode.JSON;
            generator.constraint = schema;
            generator.parameters.withGuidedJson(schema);
            String output = generator.generate(prompt);
            try {
                return JSON.readValue(output, type);
            } catch (JsonProcessingException e) {
                throw new SketchesGenerationException("Generated output was not valid JSON for " + type, e);
            }
        }
    }

    private enum Mode {
        NONE,
        REGEX,
        CHOICE,
        JSON
    }
}
