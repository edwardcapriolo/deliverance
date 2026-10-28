package io.teknek.sketches;

import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.CausalLanguageModel;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class SketchesGeneratorTest {
    record Review(String sentiment, int score) {
    }

    @Test
    void rawGenerationReturnsModelTextWithoutGuidance() {
        CausalLanguageModel model = modelReturning("plain output");

        String result = Sketches.generator(model).raw("Say something");

        assertEquals("plain output", result);
        ArgumentCaptor<GeneratorParameters> parameters = parametersCaptor();
        verify(model).generate(any(UUID.class), any(PromptContext.class), parameters.capture(), any(GenerateEvent.class));
        assertTrue(parameters.getValue().guidedChoice.isEmpty());
        assertTrue(parameters.getValue().guidedRegex.isEmpty());
        assertTrue(parameters.getValue().guidedJson.isEmpty());
    }

    @Test
    void regexGenerationForwardsRegexConstraint() {
        CausalLanguageModel model = modelReturning("AB-1234");

        String result = Sketches.generator(model)
                .regex("[A-Z]{2}-[0-9]{4}")
                .generate("Create a ticket number");

        assertEquals("AB-1234", result);
        ArgumentCaptor<GeneratorParameters> parameters = parametersCaptor();
        verify(model).generate(any(UUID.class), any(PromptContext.class), parameters.capture(), any(GenerateEvent.class));
        assertEquals("[A-Z]{2}-[0-9]{4}", parameters.getValue().guidedRegex.orElseThrow());
    }

    @Test
    void choiceGenerationForwardsChoices() {
        CausalLanguageModel model = modelReturning("Positive");

        String result = Sketches.generator(model)
                .choice(List.of("Positive", "Negative", "Neutral"))
                .generate("Classify this review");

        assertEquals("Positive", result);
        ArgumentCaptor<GeneratorParameters> parameters = parametersCaptor();
        verify(model).generate(any(UUID.class), any(PromptContext.class), parameters.capture(), any(GenerateEvent.class));
        assertEquals(List.of("Positive", "Negative", "Neutral"), parameters.getValue().guidedChoice.orElseThrow());
    }

    @Test
    void jsonGenerationReturnsTypedRecord() {
        CausalLanguageModel model = modelReturning("{\"sentiment\":\"Positive\",\"score\":9}");

        Review result = Sketches.generator(model)
                .json(Review.class)
                .generate("Analyze this review");

        assertEquals(new Review("Positive", 9), result);
        ArgumentCaptor<GeneratorParameters> parameters = parametersCaptor();
        verify(model).generate(any(UUID.class), any(PromptContext.class), parameters.capture(), any(GenerateEvent.class));
        assertTrue(parameters.getValue().guidedJson.orElseThrow().contains("sentiment"));
    }

    @Test
    void jsonGenerationSupportsGenericTypeReferences() {
        CausalLanguageModel model = modelReturning("[{\"sentiment\":\"Positive\",\"score\":9}]");

        List<Review> result = Sketches.generator(model)
                .json(new com.fasterxml.jackson.core.type.TypeReference<List<Review>>() {})
                .generate("Analyze these reviews");

        assertEquals(List.of(new Review("Positive", 9)), result);
    }

    @Test
    void invalidJsonBecomesSketchesGenerationException() {
        CausalLanguageModel model = modelReturning("not-json");

        SketchesGenerationException error = assertThrows(SketchesGenerationException.class, () ->
                Sketches.generator(model).json(Review.class).generate("Analyze this review"));

        assertTrue(error.getMessage().contains("valid JSON"));
    }

    @Test
    void modelFailureBecomesSketchesGenerationException() {
        CausalLanguageModel model = mock(CausalLanguageModel.class);
        when(model.promptSupport()).thenReturn(Optional.empty());
        when(model.generate(any(UUID.class), any(PromptContext.class), any(GeneratorParameters.class), any(GenerateEvent.class)))
                .thenThrow(new IllegalStateException("model failed"));

        SketchesGenerationException error = assertThrows(SketchesGenerationException.class, () ->
                Sketches.generator(model).raw("prompt"));

        assertEquals("Guided generation failed", error.getMessage());
        assertEquals("model failed", error.getCause().getMessage());
    }

    private static CausalLanguageModel modelReturning(String text) {
        CausalLanguageModel model = mock(CausalLanguageModel.class);
        when(model.promptSupport()).thenReturn(Optional.empty());
        when(model.generate(any(UUID.class), any(PromptContext.class), any(GeneratorParameters.class), any(GenerateEvent.class)))
                .thenReturn(new Response(text, text, FinishReason.STOP_TOKEN, 0, List.of(), 0, 0, List.of()));
        return model;
    }

    private static ArgumentCaptor<GeneratorParameters> parametersCaptor() {
        return ArgumentCaptor.forClass(GeneratorParameters.class);
    }
}
