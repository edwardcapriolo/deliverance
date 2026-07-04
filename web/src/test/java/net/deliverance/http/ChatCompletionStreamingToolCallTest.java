package net.deliverance.http;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.ChatCompletionRequestMessage;
import io.teknek.deliverance.model.ChatCompletionRequestUserMessage;
import io.teknek.deliverance.model.ChatCompletionRequestUserMessageContent;
import io.teknek.deliverance.model.ChatCompletionTool;
import io.teknek.deliverance.model.CausalLanguageModel;
import io.teknek.deliverance.model.CreateChatCompletionRequest;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.model.ReasoningFieldNames;
import io.teknek.deliverance.nanocode.NanocodeDeliverance;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.toolcallparser.QwenToolCallParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.Mockito;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CancellationException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;

class ChatCompletionStreamingToolCallTest {

    @TempDir
    Path tempDir;

    @Test
    void streamingQwenToolCallSuppressesRawMarkupAndNanocodeRunsGrep() throws Exception {
        Path file = tempDir.resolve("ed");
        Files.writeString(file, "dog\ncat\nbird\n");
        String generated = "assistant <think>checking</think>\n"
                + "<tool_call>"
                + JsonUtils.om.writeValueAsString(Map.of("name", "grep", "arguments",
                        Map.of("path", file.toString(), "pattern", "cat", "limit", 10)))
                + "</tool_call>"
                + "<tool_call>"
                + JsonUtils.om.writeValueAsString(Map.of("name", "grep", "arguments",
                        Map.of("path", file.toString(), "pattern", "cat", "limit", 10)))
                + "</tool_call>";

        CausalLanguageModel model = Mockito.mock(CausalLanguageModel.class);
        Mockito.when(model.promptSupport()).thenReturn(Optional.of(new PromptSupport(Map.of("default", "prompt"), "", true)));
        Mockito.when(model.getToolCallParser()).thenReturn(new QwenToolCallParser());
        Mockito.when(model.generate(any(UUID.class), any(), any(), any(GenerateEvent.class))).thenAnswer(invocation -> {
            GenerateEvent event = invocation.getArgument(3);
            emitChunks(event, generated);
            return new Response("", generated, FinishReason.TOOL_CALLS, 0, List.of(), 0, 0, List.of());
        });

        CapturingController controller = new CapturingController();
        MultiModelConfig config = new MultiModelConfig();
        config.setModelName("test-model");
        config.setModelOwner("test-owner");
        ReflectionTestUtils.setField(controller, "models", Map.of(config, model));

        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model("test-model")
                .stream(true)
                .maxTokens(128)
                .messages(List.of(new ChatCompletionRequestMessage(new ChatCompletionRequestUserMessage()
                        .content(new ChatCompletionRequestUserMessageContent("grep the file " + file + " for cat")))))
                .tools(nanocodeTools());

        Object response = controller.createChatCompletion(Map.of(), request);

        assertTrue(response instanceof SseEmitter);
        assertTrue(controller.emitter.awaitComplete(), "stream did not complete");
        String streamText = controller.emitter.events.toString();
        assertFalse(streamText.contains("<tool_call>"), "raw Qwen tool markup leaked into stream");

        ObjectNode toolDelta = controller.emitter.events.stream()
                .filter(ObjectNode.class::isInstance)
                .map(ObjectNode.class::cast)
                .filter(node -> node.path("choices").path(0).path("delta").path("tool_calls").isArray())
                .findFirst()
                .orElseThrow();
        JsonNode call = toolDelta.path("choices").path(0).path("delta").path("tool_calls").path(0);
        assertEquals("grep", call.path("function").path("name").asText());
        JsonNode arguments = JsonUtils.om.readTree(call.path("function").path("arguments").asText());

        String grepResult = NanocodeDeliverance.executeTool("grep", arguments);

        assertTrue(grepResult.contains("matches=1"));
        assertTrue(grepResult.contains(file + ":2:cat"));
        assertEquals(1, controller.emitter.events.stream()
                .filter(ObjectNode.class::isInstance)
                .map(ObjectNode.class::cast)
                .filter(node -> node.path("choices").path(0).path("delta").path("tool_calls").isArray())
                .count(), "duplicate raw tool calls should collapse to one structured tool delta");
    }

    @Test
    void streamingQwenToolCallParsesAccumulatedStreamWhenFinalResponseOmitsToolMarkup() throws Exception {
        Path file = tempDir.resolve("README.md");
        Files.writeString(file, "java\npython\n");
        String generated = "<think>call grep</think>\n"
                + "<tool_call>"
                + JsonUtils.om.writeValueAsString(Map.of("name", "grep", "arguments",
                        Map.of("path", file.toString(), "pattern", "java", "limit", 10)))
                + "</tool_call>";

        CausalLanguageModel model = Mockito.mock(CausalLanguageModel.class);
        Mockito.when(model.promptSupport()).thenReturn(Optional.of(new PromptSupport(Map.of("default", "prompt"), "", true)));
        Mockito.when(model.getToolCallParser()).thenReturn(new QwenToolCallParser());
        Mockito.when(model.generate(any(UUID.class), any(), any(), any(GenerateEvent.class))).thenAnswer(invocation -> {
            GenerateEvent event = invocation.getArgument(3);
            emitChunks(event, generated);
            return new Response("", "<think>call grep</think>\n", FinishReason.TOOL_CALLS,
                    0, List.of(), 0, 0, List.of());
        });

        CapturingController controller = new CapturingController();
        MultiModelConfig config = new MultiModelConfig();
        config.setModelName("test-model");
        config.setModelOwner("test-owner");
        ReflectionTestUtils.setField(controller, "models", Map.of(config, model));

        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model("test-model")
                .stream(true)
                .maxTokens(128)
                .messages(List.of(new ChatCompletionRequestMessage(new ChatCompletionRequestUserMessage()
                        .content(new ChatCompletionRequestUserMessageContent("grep the file " + file + " for java")))))
                .tools(nanocodeTools());

        Object response = controller.createChatCompletion(Map.of(), request);

        assertTrue(response instanceof SseEmitter);
        assertTrue(controller.emitter.awaitComplete(), "stream did not complete");
        String streamText = controller.emitter.events.toString();
        assertFalse(streamText.contains("<tool_call>"), "raw Qwen tool markup leaked into stream");
        ObjectNode toolDelta = controller.emitter.events.stream()
                .filter(ObjectNode.class::isInstance)
                .map(ObjectNode.class::cast)
                .filter(node -> node.path("choices").path(0).path("delta").path("tool_calls").isArray())
                .findFirst()
                .orElseThrow();

        JsonNode call = toolDelta.path("choices").path(0).path("delta").path("tool_calls").path(0);
        assertEquals("grep", call.path("function").path("name").asText());
        JsonNode arguments = JsonUtils.om.readTree(call.path("function").path("arguments").asText());
        assertEquals(file.toString(), arguments.path("path").asText());
        assertEquals("java", arguments.path("pattern").asText());
    }

    @Test
    void streamSendFailureCancelsGenerationCallbackLoop() throws Exception {
        CausalLanguageModel model = Mockito.mock(CausalLanguageModel.class);
        Mockito.when(model.promptSupport()).thenReturn(Optional.of(new PromptSupport(Map.of("default", "prompt"), "", true)));
        Mockito.when(model.getToolCallParser()).thenReturn(new QwenToolCallParser());
        AtomicInteger emitted = new AtomicInteger();
        AtomicReference<Throwable> stoppedBy = new AtomicReference<>();
        CountDownLatch stopped = new CountDownLatch(1);
        Mockito.when(model.generate(any(UUID.class), any(), any(), any(GenerateEvent.class))).thenAnswer(invocation -> {
            GenerateEvent event = invocation.getArgument(3);
            try {
                for (int i = 0; i < 100; i++) {
                    emitted.incrementAndGet();
                    event.emit(i, "x", "x", 0.0f);
                }
            } catch (Throwable t) {
                stoppedBy.set(t);
                stopped.countDown();
                throw t;
            }
            stopped.countDown();
            return new Response("", "", FinishReason.STOP_TOKEN, 0, List.of(), 0, 0, List.of());
        });

        DisconnectingController controller = new DisconnectingController();
        MultiModelConfig config = new MultiModelConfig();
        config.setModelName("test-model");
        config.setModelOwner("test-owner");
        ReflectionTestUtils.setField(controller, "models", Map.of(config, model));
        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model("test-model")
                .stream(true)
                .maxTokens(128)
                .messages(List.of(new ChatCompletionRequestMessage(new ChatCompletionRequestUserMessage()
                        .content(new ChatCompletionRequestUserMessageContent("hello")))));

        Object response = controller.createChatCompletion(Map.of(), request);

        assertTrue(response instanceof SseEmitter);
        assertTrue(stopped.await(5, TimeUnit.SECONDS), "generation did not stop after client disconnect");
        assertTrue(stoppedBy.get() instanceof CancellationException, "expected cancellation, got " + stoppedBy.get());
        assertTrue(emitted.get() < 100, "generation kept emitting after stream closed");
        assertEquals(1, controller.emitter.sends.get());
    }

    @Test
    void streamingReasoningUsesReasoningContentAndDoesNotLeakIntoContent() throws Exception {
        CausalLanguageModel model = Mockito.mock(CausalLanguageModel.class);
        Mockito.when(model.promptSupport()).thenReturn(Optional.of(new PromptSupport(Map.of("default", "prompt"), "", true)));
        Mockito.when(model.getToolCallParser()).thenReturn(new QwenToolCallParser());
        Mockito.when(model.generate(any(UUID.class), any(), any(), any(GenerateEvent.class))).thenAnswer(invocation -> {
            GenerateEvent event = invocation.getArgument(3);
            for (String chunk : List.of("<", "think>", "reason", "</", "think>", "answer")) {
                event.emit(0, chunk, chunk, 0.0f);
            }
            return new Response("answer", "<think>reason</think>answer", FinishReason.STOP_TOKEN, 0, List.of(), 0, 0, List.of())
                    .copyWithText("answer", "<think>reason</think>answer", "reason");
        });

        CapturingController controller = new CapturingController();
        MultiModelConfig config = new MultiModelConfig();
        config.setModelName("test-model");
        config.setModelOwner("test-owner");
        ReflectionTestUtils.setField(controller, "models", Map.of(config, model));
        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model("test-model")
                .stream(true)
                .maxTokens(128)
                .messages(List.of(new ChatCompletionRequestMessage(new ChatCompletionRequestUserMessage()
                        .content(new ChatCompletionRequestUserMessageContent("hello")))));

        Object response = controller.createChatCompletion(Map.of(), request);

        assertTrue(response instanceof SseEmitter);
        assertTrue(controller.emitter.awaitComplete(), "stream did not complete");
        String streamText = controller.emitter.events.toString();
        assertTrue(streamText.contains(ReasoningFieldNames.OPENAI));
        assertTrue(streamText.contains("answer"));
        assertFalse(streamText.contains("<think>"));
    }

    @Test
    void reasoningStreamSplitterConsumesSplitClosingThinkTag() {
        ChatCompletionController.ReasoningStreamSplitter splitter = new ChatCompletionController.ReasoningStreamSplitter();

        assertEquals("", splitter.accept("<think>").content());
        assertEquals("reason", splitter.accept("reason</").reasoning());
        assertEquals("", splitter.accept("think").reasoning());
        ChatCompletionController.ReasoningStreamPart afterClose = splitter.accept(">answer");

        assertEquals("", afterClose.reasoning());
        assertEquals("answer", afterClose.content());
    }

    private static void emitChunks(GenerateEvent event, String generated) {
        int token = 0;
        for (String chunk : List.of(
                generated.substring(0, generated.indexOf("<tool_call>")),
                "<", "tool", "_call>",
                generated.substring(generated.indexOf("<tool_call>") + "<tool_call>".length()))) {
            event.emit(token++, chunk, chunk, 0.0f);
        }
    }

    private static List<ChatCompletionTool> nanocodeTools() {
        return NanocodeDeliverance.defaultToolSchema(false).stream()
                .map(tool -> JsonUtils.om.convertValue(tool, ChatCompletionTool.class))
                .toList();
    }

    private static class CapturingController extends ChatCompletionController {
        private final CapturingSseEmitter emitter = new CapturingSseEmitter();

        private CapturingController() {
            super(Optional.empty(), false, "off", "openai", false);
        }

        @Override
        protected SseEmitter newSseEmitter() {
            return emitter;
        }
    }

    private static class CapturingSseEmitter extends SseEmitter {
        private final List<Object> events = new ArrayList<>();
        private final CountDownLatch complete = new CountDownLatch(1);

        private CapturingSseEmitter() {
            super(-1L);
        }

        @Override
        public synchronized void send(Object object) throws IOException {
            events.add(object);
        }

        @Override
        public void complete() {
            complete.countDown();
        }

        @Override
        public void completeWithError(Throwable ex) {
            complete.countDown();
        }

        private boolean awaitComplete() throws InterruptedException {
            return complete.await(5, TimeUnit.SECONDS);
        }
    }

    private static class DisconnectingController extends ChatCompletionController {
        private final DisconnectingSseEmitter emitter = new DisconnectingSseEmitter();

        private DisconnectingController() {
            super(Optional.empty(), false, "off", "openai", false);
        }

        @Override
        protected SseEmitter newSseEmitter() {
            return emitter;
        }
    }

    private static class DisconnectingSseEmitter extends SseEmitter {
        private final AtomicInteger sends = new AtomicInteger();

        private DisconnectingSseEmitter() {
            super(-1L);
        }

        @Override
        public synchronized void send(Object object) throws IOException {
            sends.incrementAndGet();
            throw new IOException("client disconnected");
        }
    }
}
