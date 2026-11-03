package io.teknek.reckoning.web;

import com.fasterxml.jackson.databind.JsonNode;
import io.teknek.reckoning.JsonUtils;
import io.teknek.reckoning.generator.FinishReason;
import io.teknek.reckoning.generator.Response;
import io.teknek.reckoning.model.CausalLanguageModel;
import io.teknek.reckoning.model.GenerateEvent;
import io.teknek.reckoning.nanocode.NanocodeReckoning;
import io.teknek.reckoning.safetensors.prompt.PromptContext;
import io.teknek.reckoning.safetensors.prompt.PromptSupport;
import io.teknek.reckoning.toolcallparser.QwenToolCallParser;
import net.reckoning.http.ReckoningApplication;
import net.reckoning.http.MultiModelConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;

@SpringBootTest(classes = {ReckoningApplication.class, NanocodeStreamingE2ETest.TestBeans.class},
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = {
                "spring.main.allow-bean-definition-overriding=true",
                "reckoning.tensor.operations.type=jvector"
        })
class NanocodeStreamingE2ETest {

    @LocalServerPort
    int port;

    @TempDir
    Path tempDir;

    @Test
    void nanocodeStreamsToolCallRunsGrepAndSendsToolResultBackToServer() throws Exception {
        Path file = tempDir.resolve("ed");
        Files.writeString(file, "dog\ncat\nbird\n");
        AtomicInteger generateCalls = new AtomicInteger();
        AtomicReference<String> firstPrompt = new AtomicReference<>();
        AtomicReference<String> secondPrompt = new AtomicReference<>();

        Mockito.reset(TestBeans.MODEL);
        stubQwenModel(TestBeans.MODEL, file, generateCalls, firstPrompt, secondPrompt);
        NanocodeReckoning.ToolExecutor toolExecutor = Mockito.mock(NanocodeReckoning.ToolExecutor.class);
        Mockito.when(toolExecutor.run(eq("grep"), any(JsonNode.class)))
                .thenAnswer(invocation -> NanocodeReckoning.executeTool(invocation.getArgument(0), invocation.getArgument(1)));
        NanocodeReckoning nanocode = new NanocodeReckoning(new NanocodeReckoning.Config(
                "http://localhost:" + port,
                "test-qwen",
                null,
                256,
                2000,
                3,
                0.0d,
                true,
                false,
                true,
                "eclipse-temurin:25-jdk",
                true,
                "You are a concise coding assistant.",
                "small",
                Map.of("small", "Reason briefly. Keep reasoning under 2 sentences."),
                false), toolExecutor);
        List<Map<String, Object>> messages = new ArrayList<>();
        messages.add(NanocodeReckoning.message("user", "grep the file " + file + " for cat"));

        nanocode.runConversationTurn(messages, tempDir.toString());

        assertEquals(2, generateCalls.get(), "nanocode should call model once for tool call and once after tool result");
        assertTrue(firstPrompt.get().contains("# Tools"), firstPrompt.get());
        assertTrue(firstPrompt.get().contains("<tool_call>"), firstPrompt.get());
        assertTrue(firstPrompt.get().contains("grep the file " + file + " for cat"), firstPrompt.get());
        assertTrue(secondPrompt.get().contains("<tool_response>"), secondPrompt.get());
        assertTrue(secondPrompt.get().contains("matches=1"), secondPrompt.get());
        assertTrue(secondPrompt.get().contains(file + ":2:cat"), secondPrompt.get());
        ArgumentCaptor<JsonNode> arguments = ArgumentCaptor.forClass(JsonNode.class);
        Mockito.verify(toolExecutor).run(eq("grep"), arguments.capture());
        assertEquals(file.toString(), arguments.getValue().path("path").asText());
        assertEquals("cat", arguments.getValue().path("pattern").asText());
        assertTrue(messages.stream().anyMatch(message -> "tool".equals(message.get("role"))
                && String.valueOf(message.get("content")).contains("matches=1")));
    }

    private static void stubQwenModel(CausalLanguageModel model, Path file, AtomicInteger generateCalls,
            AtomicReference<String> firstPrompt, AtomicReference<String> secondPrompt) {
        Mockito.when(model.promptSupport()).thenReturn(Optional.of(new PromptSupport(
                Map.of("default", "messages[::-1] <think>"), "", true)));
        Mockito.when(model.getToolCallParser()).thenReturn(new QwenToolCallParser());
        Mockito.when(model.generate(any(UUID.class), any(PromptContext.class), any(), any(GenerateEvent.class)))
                .thenAnswer(invocation -> {
                    PromptContext promptContext = invocation.getArgument(1);
                    GenerateEvent event = invocation.getArgument(3);
                    int call = generateCalls.incrementAndGet();
                    if (call == 1) {
                        firstPrompt.set(promptContext.getPrompt());
                        String toolCall = "<think>Use grep.</think>\n<tool_call>"
                                + JsonUtils.om.writeValueAsString(Map.of("name", "grep", "arguments",
                                        Map.of("path", file.toString(), "pattern", "cat", "limit", 10)))
                                + "</tool_call>";
                        emitChunks(event, toolCall);
                        return new Response("", toolCall, FinishReason.TOOL_CALLS, 0, List.of(), 0, 0, List.of());
                    }
                    secondPrompt.set(promptContext.getPrompt());
                    String finalText = "The grep result contains cat.";
                    event.emit(0, finalText, finalText, 0.0f);
                    return new Response(finalText, finalText, FinishReason.STOP_TOKEN, 0, List.of(), 0, 0, List.of());
                });
    }

    private static void emitChunks(GenerateEvent event, String generated) {
        int token = 0;
        int toolStart = generated.indexOf("<tool_call>");
        for (String chunk : List.of(
                generated.substring(0, toolStart),
                "<",
                "tool",
                "_call>",
                generated.substring(toolStart + "<tool_call>".length()))) {
            event.emit(token++, chunk, chunk, 0.0f);
        }
    }

    @TestConfiguration
    static class TestBeans {
        private static final CausalLanguageModel MODEL = Mockito.mock(CausalLanguageModel.class);

        @Bean("causalLanguageModels")
        @Primary
        Map<MultiModelConfig, CausalLanguageModel> causalLanguageModels() {
            MultiModelConfig config = new MultiModelConfig();
            config.setModelName("test-qwen");
            config.setModelOwner("test");
            return Map.of(config, MODEL);
        }

        @Bean("embeddingModels")
        @Primary
        Map<MultiModelConfig, io.teknek.reckoning.model.AbstractModel> embeddingModels() {
            return Map.of();
        }
    }
}
