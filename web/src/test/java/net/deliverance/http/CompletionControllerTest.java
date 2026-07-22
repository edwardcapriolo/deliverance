package net.deliverance.http;

import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.CausalLanguageModel;
import io.teknek.deliverance.model.CreateCompletionRequest;
import io.teknek.deliverance.model.CreateCompletionResponse;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;
import org.springframework.http.ResponseEntity;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;

class CompletionControllerTest {

    @Test
    void completionUsesRawPromptWithoutPromptSupport() {
        CausalLanguageModel model = Mockito.mock(CausalLanguageModel.class);
        Mockito.when(model.generate(any(UUID.class), any(), any(), any(GenerateEvent.class)))
                .thenReturn(new Response(" raw completion", " raw completion", FinishReason.STOP_TOKEN,
                        0, List.of(), 0, 0, List.of()));
        Mockito.when(model.promptSupport()).thenReturn(Optional.empty());
        ChatCompletionController controller = controller(model);

        Object response = controller.createCompletion(Map.of(), completionRequest(false));

        assertTrue(response instanceof ResponseEntity<?>);
        Object body = ((ResponseEntity<?>) response).getBody();
        assertTrue(body instanceof CreateCompletionResponse);
        CreateCompletionResponse completion = (CreateCompletionResponse) body;
        assertEquals("text_completion", completion.getObject().getValue());
        assertEquals("test-model", completion.getModel());
        assertEquals(" raw completion", completion.getChoices().getFirst().getText());
        assertEquals("stop", completion.getChoices().getFirst().getFinishReason().getValue());

        ArgumentCaptor<PromptContext> prompt = ArgumentCaptor.forClass(PromptContext.class);
        Mockito.verify(model).generate(any(UUID.class), prompt.capture(), any(), any(GenerateEvent.class));
        assertEquals("RAW ANTARES PROMPT", prompt.getValue().getPrompt());
        Mockito.verify(model, Mockito.never()).promptSupport();
    }

    @Test
    void streamingCompletionEmitsLegacyTextCompletionEvents() throws Exception {
        CausalLanguageModel model = Mockito.mock(CausalLanguageModel.class);
        Mockito.when(model.generate(any(UUID.class), any(), any(), any(GenerateEvent.class))).thenAnswer(invocation -> {
            GenerateEvent event = invocation.getArgument(3);
            event.emit(1, "hello", "hello", 0.0f);
            event.emit(2, " world", " world", 0.0f);
            return new Response("hello world", "hello world", FinishReason.STOP_TOKEN, 0, List.of(), 0, 0,
                    List.of());
        });
        CapturingController controller = new CapturingController(model);

        Object response = controller.createCompletion(Map.of(), completionRequest(true));

        assertTrue(response instanceof SseEmitter);
        assertTrue(controller.emitter.awaitComplete(), "stream did not complete");
        String streamText = controller.emitter.events.toString();
        assertTrue(streamText.contains("hello"), streamText);
        assertTrue(streamText.contains("text_completion"), streamText);
        assertTrue(streamText.contains("[DONE]"), streamText);
    }

    private static CreateCompletionRequest completionRequest(boolean stream) {
        return new CreateCompletionRequest()
                .model("test-model")
                .prompt("RAW ANTARES PROMPT")
                .maxTokens(16)
                .temperature(BigDecimal.valueOf(0.3))
                .stream(stream);
    }

    private static ChatCompletionController controller(CausalLanguageModel model) {
        ChatCompletionController controller = new ChatCompletionController(Optional.empty(), false, "off", "openai", false);
        MultiModelConfig config = new MultiModelConfig();
        config.setModelName("test-model");
        config.setModelOwner("test-owner");
        ReflectionTestUtils.setField(controller, "models", Map.of(config, model));
        return controller;
    }

    private static class CapturingController extends ChatCompletionController {
        private final CapturingSseEmitter emitter = new CapturingSseEmitter();

        private CapturingController(CausalLanguageModel model) {
            super(Optional.empty(), false, "off", "openai", false);
            MultiModelConfig config = new MultiModelConfig();
            config.setModelName("test-model");
            config.setModelOwner("test-owner");
            ReflectionTestUtils.setField(this, "models", Map.of(config, model));
        }

        @Override
        protected SseEmitter newSseEmitter() {
            return emitter;
        }
    }

    private static class CapturingSseEmitter extends SseEmitter {
        private final List<Object> events = new java.util.ArrayList<>();
        private final CountDownLatch complete = new CountDownLatch(1);

        private CapturingSseEmitter() {
            super(-1L);
        }

        @Override
        public synchronized void send(Object object) throws IOException {
            events.add(object);
        }

        @Override
        public synchronized void send(SseEventBuilder builder) throws IOException {
            events.add(builder.toString());
        }

        @Override
        public void complete() {
            complete.countDown();
        }

        @Override
        public void completeWithError(Throwable ex) {
            events.add(ex.toString());
            complete.countDown();
        }

        private boolean awaitComplete() throws InterruptedException {
            return complete.await(5, TimeUnit.SECONDS);
        }
    }
}
