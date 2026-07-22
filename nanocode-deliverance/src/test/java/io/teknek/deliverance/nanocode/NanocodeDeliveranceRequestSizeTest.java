package io.teknek.deliverance.nanocode;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.teknek.deliverance.client.model.CreateChatCompletionRequest;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NanocodeDeliveranceRequestSizeTest {
    private static final ObjectMapper JSON = NanocodeDeliverance.clientMapper();

    @Test
    @SuppressWarnings({"rawtypes", "unchecked"})
    void defaultToolsProduceSmallChatRequest() throws Exception {
        NanocodeDeliverance agent = new NanocodeDeliverance(new NanocodeDeliverance.Config(
                "http://localhost:8085", "Llama-3.2-3B-Instruct-JQ4", null, 256, 2000, 3, 0.0d, true, false, true,
                "eclipse-temurin:25-jdk", true, "You are a concise coding assistant.", "small",
                Map.of("small", "Reason briefly. Keep reasoning under 2 sentences."), false));
        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model("Llama-3.2-3B-Instruct-JQ4")
                .maxTokens(256)
                .temperature(BigDecimal.ZERO)

                .messages((List) List.of(
                        NanocodeDeliverance.message("system", "You are a concise coding assistant. cwd: /tmp."),
                        NanocodeDeliverance.message("user", "hi")))
                .tools(agent.toolSchema())
                .parallelToolCalls(false);

        String json = JSON.writeValueAsString(request);
        int bytes = json.getBytes(StandardCharsets.UTF_8).length;
        System.out.println("nanocode request bytes=" + bytes);
        System.out.println(json);
        assertTrue(bytes < 20_000, "expected tool request to remain small, bytes=" + bytes);
    }

    @Test
    void systemPromptTellsModelToCallToolsInsteadOfDescribingJson() {
        String prompt = NanocodeDeliverance.systemPrompt("/tmp/work");

        assertTrue(prompt.contains("concise coding assistant"));
        assertTrue(prompt.contains("cwd: /tmp/work"));
    }

    @Test
    void defaultToolsIncludeWebFetch() {
        NanocodeDeliverance agent = new NanocodeDeliverance(new NanocodeDeliverance.Config(
                "http://localhost:8085", "test", null, 256, 2000, 3, 0.0d, true, false, true,
                "eclipse-temurin:25-jdk", true, "You are a concise coding assistant.", "small",
                Map.of("small", "Reason briefly. Keep reasoning under 2 sentences."), false));

        assertTrue(agent.toolSchema().stream()
                .anyMatch(tool -> "web_fetch".equals(tool.getFunction().getName())));
    }

    @Test
    void terminalAliasIsAdvertisedOnlyWhenRiskyToolsAreEnabled() {
        NanocodeDeliverance safeAgent = new NanocodeDeliverance(new NanocodeDeliverance.Config(
                "http://localhost:8085", "test", null, 256, 2000, 3, 0.0d, true, false, true,
                "eclipse-temurin:25-jdk", true, "You are a concise coding assistant.", "small",
                Map.of("small", "Reason briefly. Keep reasoning under 2 sentences."), false));
        NanocodeDeliverance riskyAgent = new NanocodeDeliverance(new NanocodeDeliverance.Config(
                "http://localhost:8085", "test", null, 256, 2000, 3, 0.0d, true, true, true,
                "eclipse-temurin:25-jdk", true, "You are a concise coding assistant.", "small",
                Map.of("small", "Reason briefly. Keep reasoning under 2 sentences."), false));

        assertTrue(safeAgent.toolSchema().stream()
                .noneMatch(tool -> "terminal".equals(tool.getFunction().getName())));
        assertTrue(riskyAgent.toolSchema().stream()
                .anyMatch(tool -> "terminal".equals(tool.getFunction().getName())));
    }

    @Test
    void terminalAliasRoutesToShellExecutorWhenEnabled() throws Exception {
        String result = NanocodeDeliverance.runToolForTest("terminal", JSON.readTree("{\"command\":\"printf antares\"}"),
                true, null);

        assertEquals("antares", result);
    }

    @Test
    void webFetchRejectsNonPublicUrls() throws Exception {
        assertTrue(NanocodeDeliverance.executeTool("web_fetch", JSON.readTree("{\"url\":\"file:///etc/passwd\"}"))
                .contains("supports only http and https"));
        assertTrue(NanocodeDeliverance.executeTool("web_fetch", JSON.readTree("{\"url\":\"http://localhost:8080\"}"))
                .contains("only fetches public"));
    }

    @Test
    void configLoadsFromSingleJsonFile() throws Exception {
        Path config = Files.createTempFile("nanocode", ".json");
        Files.writeString(config, """
                {
                  "baseUrl": "http://localhost:8085/",
                  "model": "Qwen3-4B-JQ4",
                  "maxTokens": 512,
                  "maxToolRounds": 5,
                  "enableThinking": false
                }
                """);

        NanocodeDeliverance.Config loaded = NanocodeDeliverance.Config.parse(new String[] {"--config", config.toString()});

        assertEquals("http://localhost:8085", loaded.baseUrl());
        assertEquals("Qwen3-4B-JQ4", loaded.model());
        assertEquals(512, loaded.maxTokens());
        assertEquals(5, loaded.maxToolRounds());
        assertEquals(2000, loaded.maxToolResultChars());
        assertTrue(loaded.toolsEnabled());
        assertTrue(!loaded.enableThinking());
    }

    @Test
    void configCommandSetsRoundsAndThinking() {
        NanocodeDeliverance agent = new NanocodeDeliverance(new NanocodeDeliverance.Config(
                "http://localhost:8085", "test", null, 256, 2000, 3, 0.0d, true, false, true,
                "eclipse-temurin:25-jdk", true, "You are a concise coding assistant.", "small",
                Map.of("small", "Reason briefly. Keep reasoning under 2 sentences."), false));

        assertTrue(agent.handleConfigCommand("/config set rounds 5"));
        assertTrue(agent.handleConfigCommand("/config set thinking off"));
        assertTrue(agent.handleConfigCommand("/config get rounds"));
        assertTrue(agent.handleConfigCommand("/config get thinking"));
    }

    @Test
    void configRejectsOldSwitches() {
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> NanocodeDeliverance.Config.parse(new String[] {"--model", "Qwen3-4B-JQ4"}));

        assertTrue(error.getMessage().contains("--config"));
    }

    @Test
    void reasoningDeltaReadsOpenAiAndVllmFields() throws Exception {
        assertEquals("openai", NanocodeDeliverance.reasoningDelta(JSON.readTree("{\"" + NanocodeDeliverance.OPENAI_REASONING_FIELD + "\":\"openai\"}")));
        assertEquals("vllm", NanocodeDeliverance.reasoningDelta(JSON.readTree("{\"" + NanocodeDeliverance.VLLM_REASONING_FIELD + "\":\"vllm\"}")));
    }

    @Test
    void assistantMessageForNextRoundDoesNotIncludeReasoning() {
        var responseMessage = NanocodeDeliverance.streamingMessage("final answer", "hidden reasoning", List.of());

        var nextRoundMessage = NanocodeDeliverance.assistantMessage(responseMessage);

        assertEquals("assistant", nextRoundMessage.get("role"));
        assertEquals("final answer", nextRoundMessage.get("content"));
        assertTrue(!nextRoundMessage.containsKey("reasoning"));
        assertTrue(!nextRoundMessage.containsKey(NanocodeDeliverance.OPENAI_REASONING_FIELD));
        assertTrue(!nextRoundMessage.toString().contains("hidden reasoning"));
    }

    @Test
    void configLoadsRootPromptAndReasoningPromptPresets() throws Exception {
        Path config = Path.of("config-qwen3-4b-jq4.json");

        NanocodeDeliverance.Config loaded = NanocodeDeliverance.Config.fromJson(config);

        assertEquals("small", loaded.reasoningPrompt());
        assertTrue(loaded.rootPrompt().contains("concise coding assistant"));
        assertTrue(loaded.reasoningPrompts().containsKey("small"));
        assertTrue(loaded.reasoningPrompts().containsKey("medium"));
        assertTrue(loaded.reasoningPrompts().containsKey("large"));
        assertTrue(loaded.reasoningPrompts().get("small").contains("under 2 sentences"));
        assertTrue(NanocodeDeliverance.systemPrompt("/tmp", loaded.systemPromptText()).contains("under 2 sentences"));
    }
}
