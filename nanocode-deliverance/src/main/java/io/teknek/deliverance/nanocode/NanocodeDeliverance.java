package io.teknek.deliverance.nanocode;

import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.teknek.deliverance.client.api.ChatApi;
import io.teknek.deliverance.client.core.ApiClient;
import io.teknek.deliverance.client.model.ChatCompletionMessageToolCall;
import io.teknek.deliverance.client.model.ChatCompletionMessageToolCallFunction;
import io.teknek.deliverance.client.model.ChatCompletionRequestMessage;
import io.teknek.deliverance.client.model.ChatCompletionResponseMessage;
import io.teknek.deliverance.client.model.ChatCompletionTool;
import io.teknek.deliverance.client.model.CreateChatCompletionRequest;
import io.teknek.deliverance.client.model.CreateChatCompletionResponse;
import io.teknek.deliverance.client.model.CreateChatCompletionResponseChoicesInner;
import io.teknek.deliverance.client.model.FunctionObject;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import retrofit2.Retrofit;
import retrofit2.Response;
import retrofit2.converter.jackson.JacksonConverterFactory;
import retrofit2.converter.scalars.ScalarsConverterFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class NanocodeDeliverance {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final String RESET = "\033[0m";
    private static final String BOLD = "\033[1m";
    private static final String DIM = "\033[2m";
    private static final String BLUE = "\033[34m";
    private static final String CYAN = "\033[36m";
    private static final String GREEN = "\033[32m";
    private static final String RED = "\033[31m";

    private final Config config;
    private final ChatApi chatApi;
    private final OkHttpClient httpClient;
    private final ToolExecutor toolExecutor;
    private final SessionConfig sessionConfig;

    public NanocodeDeliverance(Config config) {
        this(config, (name, args) -> runTool(name, args, config.allowRiskyTools, config.javaSandboxImage));
    }

    public NanocodeDeliverance(Config config, ToolExecutor toolExecutor) {
        this.config = config;
        this.toolExecutor = toolExecutor;
        this.sessionConfig = new SessionConfig(config.maxToolRounds, config.enableThinking);
        ApiClient apiClient = new ApiClient();
        apiClient.setAdapterBuilder(new Retrofit.Builder()
                .baseUrl(config.baseUrl + "/")
                .addConverterFactory(ScalarsConverterFactory.create())
                .addConverterFactory(JacksonConverterFactory.create(clientMapper())));
        apiClient.getOkBuilder().connectTimeout(Duration.ofSeconds(10));
        apiClient.getOkBuilder().readTimeout(Duration.ofMinutes(5));
        this.httpClient = apiClient.getOkBuilder().build();
        this.chatApi = apiClient.createService(ChatApi.class);
    }

    static ObjectMapper clientMapper() {
        ObjectMapper mapper = new ObjectMapper();
        mapper.setSerializationInclusion(JsonInclude.Include.NON_NULL);
        mapper.addMixIn(ChatCompletionRequestMessage.class, NoRequestMessageTypeInfo.class);
        return mapper;
    }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NONE)
    private abstract static class NoRequestMessageTypeInfo {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        if (config.help) {
            Config.printHelp();
            return;
        }
        new NanocodeDeliverance(config).run();
    }

    private void run() throws Exception {
        String cwd = Path.of(".").toAbsolutePath().normalize().toString();
        System.out.println(BOLD + "nanocode-deliverance" + RESET + " | " + DIM + config.model + " @ "
                + config.baseUrl + " | " + cwd + RESET);
        if (config.allowRiskyTools) {
            System.out.println(RED + "risk/eval tools enabled: bash" + RESET);
        }
        if (!config.toolsEnabled) {
            System.out.println(DIM + "tools disabled" + RESET);
        } else {
            System.out.println(DIM + "config: { thinking=" + (sessionConfig.enableThinking ? "on" : "off")
                    + ", rounds=" + sessionConfig.maxToolRounds + " } /config help" + RESET);
        }

        List<Map<String, Object>> messages = new ArrayList<>();
        try (BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in))) {
            while (true) {
                System.out.println(separator());
                System.out.print(BOLD + BLUE + "> " + RESET);
                String input = stdin.readLine();
                if (input == null) {
                    break;
                }
                input = input.strip();
                if (input.isEmpty()) {
                    continue;
                }
                if (input.equals("/q") || input.equalsIgnoreCase("exit")) {
                    break;
                }
                if (input.equals("/help") || input.equals("help")) {
                    printHelp();
                    continue;
                }
                if (input.equals("/c")) {
                    messages = new ArrayList<>();
                    System.out.println(GREEN + "cleared" + RESET);
                    continue;
                }
                if (handleConfigCommand(input)) {
                    continue;
                }
                messages.add(message("user", input));
                long turnStart = System.nanoTime();
                try {
                    runConversationTurn(messages, cwd);
                    printContextSummary(messages, turnStart);
                } catch (Exception e) {
                    System.out.println(RED + "error: " + e.getMessage() + RESET);
                    printContextSummary(messages, turnStart);
                }
            }
        }
    }

    boolean handleConfigCommand(String input) {
        if (!input.startsWith("/config")) {
            return false;
        }
        String[] parts = input.split("\\s+");
        if (parts.length == 1 || (parts.length == 2 && "help".equals(parts[1]))) {
            printConfigHelp();
            return true;
        }
        if (parts.length == 3 && "get".equals(parts[1])) {
            printConfigValue(parts[2]);
            return true;
        }
        if (parts.length == 4 && "set".equals(parts[1])) {
            setConfigValue(parts[2], parts[3]);
            return true;
        }
        System.out.println(RED + "usage: /config get <rounds|thinking> or /config set <rounds|thinking> <value>" + RESET);
        return true;
    }

    private void printConfigValue(String key) {
        switch (key) {
            case "rounds" -> System.out.println(GREEN + "rounds=" + sessionConfig.maxToolRounds + RESET);
            case "thinking" -> System.out.println(GREEN + "thinking=" + (sessionConfig.enableThinking ? "on" : "off") + RESET);
            default -> System.out.println(RED + "unknown config key: " + key + RESET);
        }
    }

    private void setConfigValue(String key, String value) {
        switch (key) {
            case "rounds" -> {
                int rounds;
                try {
                    rounds = Integer.parseInt(value);
                } catch (NumberFormatException e) {
                    System.out.println(RED + "rounds must be an integer" + RESET);
                    return;
                }
                if (rounds < 1) {
                    System.out.println(RED + "rounds must be >= 1" + RESET);
                    return;
                }
                sessionConfig.maxToolRounds = rounds;
                System.out.println(GREEN + "rounds=" + rounds + RESET);
            }
            case "thinking" -> {
                Boolean parsed = parseOnOff(value);
                if (parsed == null) {
                    System.out.println(RED + "thinking must be on/off or true/false" + RESET);
                    return;
                }
                sessionConfig.enableThinking = parsed;
                System.out.println(GREEN + "thinking=" + (parsed ? "on" : "off") + RESET);
            }
            default -> System.out.println(RED + "unknown config key: " + key + RESET);
        }
    }

    private static Boolean parseOnOff(String value) {
        return switch (value.toLowerCase(Locale.ROOT)) {
            case "on", "true", "yes", "1" -> true;
            case "off", "false", "no", "0" -> false;
            default -> null;
        };
    }

    private static void printConfigHelp() {
        System.out.println(DIM + "config commands:" + RESET);
        System.out.println(DIM + "  /config get rounds" + RESET);
        System.out.println(DIM + "  /config set rounds 3" + RESET);
        System.out.println(DIM + "  /config get thinking" + RESET);
        System.out.println(DIM + "  /config set thinking off" + RESET);
        System.out.println(DIM + "session commands:" + RESET);
        System.out.println(DIM + "  /c       clear session context" + RESET);
        System.out.println(DIM + "  /q       quit" + RESET);
        System.out.println(DIM + "  /help    show help" + RESET);
    }

    private static void printHelp() {
        printConfigHelp();
    }

    public void runConversationTurn(List<Map<String, Object>> messages, String cwd) throws Exception {
        int toolRound = 0;
        while (true) {
            CreateChatCompletionResponse response = chat(messages, cwd);
            if (response.getChoices() == null || response.getChoices().isEmpty()) {
                throw new IOException("Deliverance response contained no choices");
            }
            CreateChatCompletionResponseChoicesInner choice = response.getChoices().get(0);
            ChatCompletionResponseMessage responseMessage = choice.getMessage();
            if (responseMessage == null) {
                throw new IOException("Deliverance response choice contained no message");
            }
            String content = Optional.ofNullable(responseMessage.getContent()).orElse("");
            if (!content.isBlank() && !config.streamEnabled) {
                System.out.println(CYAN + "assistant" + RESET + " " + content);
            }
            messages.add(assistantMessage(responseMessage));

            List<ChatCompletionMessageToolCall> toolCalls = responseMessage.getToolCalls();
            if (toolCalls == null || toolCalls.isEmpty()) {
                return;
            }
            toolRound++;
            if (toolRound > sessionConfig.maxToolRounds) {
                System.out.println(RED + "stopped: reached max tool rounds " + sessionConfig.maxToolRounds
                        + ". The model may be stuck; rephrase, give a more specific path, or /c to clear context." + RESET);
                return;
            }
            System.out.println(DIM + "tool round " + toolRound + "/" + sessionConfig.maxToolRounds + RESET);
            for (ChatCompletionMessageToolCall toolCall : toolCalls) {
                String id = toolCall.getId();
                String name = toolCall.getFunction().getName();
                JsonNode arguments = parseJsonObject(toolCall.getFunction().getArguments());
                System.out.println(GREEN + "tool " + name + RESET + " " + preview(arguments.toString(), 80));
                String result = truncateToolResult(toolExecutor.run(name, arguments));
                System.out.println(DIM + preview(result, 120) + RESET);
                messages.add(toolMessage(id, result));
            }
        }
    }

    @FunctionalInterface
    public interface ToolExecutor {
        String run(String name, JsonNode args);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private CreateChatCompletionResponse chat(List<Map<String, Object>> messages, String cwd) throws IOException {
        if (config.streamEnabled) {
            return streamingChat(messages, cwd);
        }
        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model(config.model)
                .maxTokens(config.maxTokens)
                .temperature(BigDecimal.valueOf(config.temperature))
                .messages((List) withSystemMessage(messages, cwd))
                .chatTemplateKwargs(Map.of("enable_thinking", sessionConfig.enableThinking))
                .parallelToolCalls(false);
        if (config.ntokens != null) {
            request.ntokens(config.ntokens);
        }
        if (config.toolsEnabled) {
            request.tools(toolSchema());
        }
        Response<CreateChatCompletionResponse> response = chatApi.createChatCompletion(request).execute();
        if (!response.isSuccessful() || response.body() == null) {
            String error = response.errorBody() == null ? "" : response.errorBody().string();
            throw new IOException("Deliverance HTTP " + response.code() + ": " + error);
        }
        return response.body();
    }

    private CreateChatCompletionResponse streamingChat(List<Map<String, Object>> messages, String cwd) throws IOException {
        ObjectNode request = JSON.createObjectNode();
        request.put("model", config.model);
        request.put("max_tokens", config.maxTokens);
        request.put("temperature", config.temperature);
        request.put("parallel_tool_calls", false);
        request.put("stream", true);
        ObjectNode chatTemplateKwargs = request.putObject("chat_template_kwargs");
        chatTemplateKwargs.put("enable_thinking", sessionConfig.enableThinking);
        if (config.ntokens != null) {
            request.put("ntokens", config.ntokens);
        }
        request.set("messages", JSON.valueToTree(withSystemMessage(messages, cwd)));
        if (config.toolsEnabled) {
            request.set("tools", JSON.valueToTree(toolSchema()));
        }
        Request httpRequest = new Request.Builder()
                .url(config.baseUrl + "/chat/completions")
                .post(RequestBody.create(JSON.writeValueAsBytes(request), MediaType.get("application/json")))
                .build();
        StringBuilder content = new StringBuilder();
        String finishReason = "stop";
        List<ToolCallAccumulator> toolCalls = new ArrayList<>();
        System.out.print(CYAN + "assistant" + RESET + " ");
        try (okhttp3.Response response = httpClient.newCall(httpRequest).execute()) {
            if (!response.isSuccessful() || response.body() == null) {
                String error = response.body() == null ? "" : response.body().string();
                throw new IOException("Deliverance HTTP " + response.code() + ": " + error);
            }
            try (BufferedReader reader = new BufferedReader(response.body().charStream())) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (!line.startsWith("data:")) {
                        continue;
                    }
                    String data = line.substring("data:".length()).trim();
                    if ("[DONE]".equals(data)) {
                        break;
                    }
                    JsonNode chunk = JSON.readTree(data);
                    JsonNode choice = chunk.path("choices").path(0);
                    JsonNode finish = choice.path("finish_reason");
                    if (!finish.isMissingNode() && !finish.isNull()) {
                        finishReason = finish.asText();
                    }
                    String delta = choice.path("delta").path("content").asText("");
                    if (!delta.isEmpty()) {
                        content.append(delta);
                        System.out.print(delta);
                    }
                    accumulateToolCalls(choice.path("delta").path("tool_calls"), toolCalls);
                }
            }
        }
        System.out.println();
        ChatCompletionResponseMessage message = new ChatCompletionResponseMessage()
                .role(ChatCompletionResponseMessage.RoleEnum.ASSISTANT)
                .content(content.toString());
        if (!toolCalls.isEmpty()) {
            message.content(null);
            for (ToolCallAccumulator toolCall : toolCalls) {
                message.addToolCallsItem(toolCall.toToolCall());
            }
        } else if ("tool_calls".equals(finishReason)) {
            throw new IOException("Deliverance stream ended with finish_reason=tool_calls but sent no delta.tool_calls");
        }
        CreateChatCompletionResponseChoicesInner choice = new CreateChatCompletionResponseChoicesInner()
                .index(0)
                .message(message)
                .finishReason(toFinishReason(finishReason));
        return new CreateChatCompletionResponse().choices(List.of(choice));
    }

    private static void accumulateToolCalls(JsonNode deltas, List<ToolCallAccumulator> toolCalls) {
        if (!deltas.isArray()) {
            return;
        }
        for (JsonNode delta : deltas) {
            int index = delta.path("index").asInt();
            while (toolCalls.size() <= index) {
                toolCalls.add(new ToolCallAccumulator());
            }
            ToolCallAccumulator accumulator = toolCalls.get(index);
            if (delta.hasNonNull("id")) {
                accumulator.id = delta.get("id").asText();
            }
            JsonNode function = delta.path("function");
            if (function.hasNonNull("name")) {
                accumulator.name = function.get("name").asText();
            }
            if (function.hasNonNull("arguments")) {
                accumulator.arguments.append(function.get("arguments").asText());
            }
        }
    }

    private static final class ToolCallAccumulator {
        private String id;
        private String name;
        private final StringBuilder arguments = new StringBuilder();

        private ChatCompletionMessageToolCall toToolCall() {
            ChatCompletionMessageToolCall toolCall = new ChatCompletionMessageToolCall();
            toolCall.id(id);
            toolCall.function(new ChatCompletionMessageToolCallFunction()
                    .name(name)
                    .arguments(arguments.toString()));
            return toolCall;
        }
    }

    private static CreateChatCompletionResponseChoicesInner.FinishReasonEnum toFinishReason(String finishReason) {
        return switch (finishReason == null ? "stop" : finishReason) {
            case "length" -> CreateChatCompletionResponseChoicesInner.FinishReasonEnum.LENGTH;
            case "tool_calls" -> CreateChatCompletionResponseChoicesInner.FinishReasonEnum.TOOL_CALLS;
            default -> CreateChatCompletionResponseChoicesInner.FinishReasonEnum.STOP;
        };
    }

    private List<Map<String, Object>> withSystemMessage(List<Map<String, Object>> messages, String cwd) {
        List<Map<String, Object>> result = new ArrayList<>();
        result.add(message("system", systemPrompt(cwd)));
        result.addAll(messages);
        return result;
    }

    static String systemPrompt(String cwd) {
        return "You are a concise coding assistant. cwd: " + cwd
                + ". Use tools when needed. Prefer small, direct changes. "
                + "For file reads, grep/searches, globbing, edits, writes, or Java execution, call the matching tool; "
                + "do not describe the tool call or print JSON in prose. "
                + "After any thinking, emit the tool call immediately in the required tool-call format.";
    }

    List<ChatCompletionTool> toolSchema() {
        return defaultToolSchema(config.allowRiskyTools);
    }

    public static List<ChatCompletionTool> defaultToolSchema(boolean allowRiskyTools) {
        List<ChatCompletionTool> tools = new ArrayList<>();
        tools.add(tool("read", "Read a text file with line numbers.", schema(
                props(prop("path", "string"), prop("offset", "integer"), prop("limit", "integer")),
                array("path"))));
        tools.add(tool("write", "Write content to a file.", schema(
                props(prop("path", "string"), prop("content", "string")),
                array("path", "content"))));
        tools.add(tool("edit", "Replace text in a file. The old text must be unique unless all=true.", schema(
                props(prop("path", "string"), prop("old", "string"), prop("new", "string"), prop("all", "boolean")),
                array("path", "old", "new"))));
        tools.add(tool("glob", "Find files by glob pattern, sorted by modified time.", schema(
                props(prop("path", "string"), prop("pattern", "string")),
                array("pattern"))));
        tools.add(tool("grep", "Search text files with a Java regular expression. Use for grep/search requests. Returns at most limit matches, default 10.", schema(
                props(prop("path", "string"), prop("pattern", "string"), prop("limit", "integer")),
                array("pattern"))));
        tools.add(tool("java_sandbox", "Run Java code or Maven tests in a one-shot isolated container with no network access by default.", schema(
                props(prop("mode", "string"), prop("files", "object"), prop("mainClass", "string"),
                        prop("timeoutSeconds", "integer"), prop("maxOutputChars", "integer")),
                array("files"))));
        if (allowRiskyTools) {
            tools.add(tool("bash", "Risky/eval tool. Run a shell command with a 30 second timeout.", schema(
                    props(prop("command", "string")),
                    array("command"))));
        }
        return tools;
    }

    private static ChatCompletionTool tool(String name, String description, JsonNode parameters) {
        Map<String, Object> parameterMap = JSON.convertValue(parameters, Map.class);
        return new ChatCompletionTool().type("function")
                .function(new FunctionObject().name(name).description(description).parameters(parameterMap));
    }

    public static String executeTool(String name, JsonNode args) {
        return runTool(name, args, false, null);
    }

    private static String runTool(String name, JsonNode args, boolean allowRiskyTools, String javaSandboxImage) {
        try {
            return switch (name) {
                case "read" -> toolRead(args);
                case "write" -> toolWrite(args);
                case "edit" -> toolEdit(args);
                case "glob" -> toolGlob(args);
                case "grep" -> toolGrep(args);
                case "java_sandbox" -> JavaSandboxTool.run(args, javaSandboxImage);
                case "bash" -> allowRiskyTools ? toolBash(args) : "error: bash disabled; set allowRiskyTools=true in the config file";
                default -> "error: unknown tool " + name;
            };
        } catch (Exception e) {
            return "error: " + e.getMessage();
        }
    }

    private static String toolRead(JsonNode args) throws IOException {
        List<String> lines = Files.readAllLines(Path.of(args.path("path").asText()));
        int offset = Math.max(0, args.path("offset").asInt(0));
        int limit = args.path("limit").asInt(lines.size());
        StringBuilder out = new StringBuilder();
        for (int i = offset; i < Math.min(lines.size(), offset + limit); i++) {
            out.append(String.format(Locale.ROOT, "%4d| %s%n", i + 1, lines.get(i)));
        }
        return out.toString();
    }

    private static String toolWrite(JsonNode args) throws IOException {
        Files.writeString(Path.of(args.path("path").asText()), args.path("content").asText());
        return "ok";
    }

    private static String toolEdit(JsonNode args) throws IOException {
        Path path = Path.of(args.path("path").asText());
        String text = Files.readString(path);
        String oldText = args.path("old").asText();
        String newText = args.path("new").asText();
        if (!text.contains(oldText)) {
            return "error: old text not found";
        }
        int count = (text.length() - text.replace(oldText, "").length()) / oldText.length();
        boolean all = args.path("all").asBoolean(false);
        if (!all && count > 1) {
            return "error: old text appears " + count + " times; set all=true or use a unique old text";
        }
        String updated = all ? text.replace(oldText, newText)
                : text.replaceFirst(Pattern.quote(oldText), Matcher.quoteReplacement(newText));
        Files.writeString(path, updated);
        return "ok";
    }

    private static String toolGlob(JsonNode args) throws IOException {
        Path base = Path.of(args.path("path").asText("."));
        String pattern = args.path("pattern").asText();
        if (!Files.exists(base)) {
            return "none";
        }
        var matcher = FileSystems.getDefault().getPathMatcher("glob:" + base.resolve(pattern));
        try (var walk = Files.walk(base)) {
            List<String> files = walk.filter(Files::isRegularFile)
                    .filter(matcher::matches)
                    .sorted(Comparator.comparing((Path p) -> {
                        try {
                            return Files.getLastModifiedTime(p);
                        } catch (IOException e) {
                            return null;
                        }
                    }, Comparator.nullsLast(Comparator.reverseOrder())))
                    .map(Path::toString)
                    .limit(100)
                    .toList();
            return files.isEmpty() ? "none" : String.join("\n", files);
        }
    }

    private static String toolGrep(JsonNode args) throws IOException {
        Pattern pattern = Pattern.compile(args.path("pattern").asText());
        Path base = Path.of(args.path("path").asText("."));
        int limit = Math.max(1, args.path("limit").asInt(10));
        List<String> hits = new ArrayList<>();
        try (var walk = Files.walk(base)) {
            for (Path file : walk.filter(Files::isRegularFile).toList()) {
                if (hits.size() >= limit) {
                    break;
                }
                List<String> lines;
                try {
                    lines = Files.readAllLines(file);
                } catch (Exception e) {
                    continue;
                }
                for (int i = 0; i < lines.size() && hits.size() < limit; i++) {
                    if (pattern.matcher(lines.get(i)).find()) {
                        hits.add(file + ":" + (i + 1) + ":" + lines.get(i));
                    }
                }
            }
        }
        return hits.isEmpty() ? "matches=0" : "matches=" + hits.size() + "\n" + String.join("\n", hits);
    }

    private static String toolBash(JsonNode args) throws IOException, InterruptedException {
        Process process = new ProcessBuilder("sh", "-c", args.path("command").asText())
                .redirectErrorStream(true)
                .start();
        boolean finished = process.waitFor(30, TimeUnit.SECONDS);
        if (!finished) {
            process.destroyForcibly();
            return new String(process.getInputStream().readAllBytes()) + "\n(timed out after 30s)";
        }
        return new String(process.getInputStream().readAllBytes());
    }

    private static JsonNode parseJsonObject(String json) throws IOException {
        JsonNode node = JSON.readTree(json == null || json.isBlank() ? "{}" : json);
        return node.isObject() ? node : JSON.createObjectNode();
    }

    public static Map<String, Object> message(String role, String content) {
        Map<String, Object> message = new HashMap<>();
        message.put("role", role);
        message.put("content", content);
        return message;
    }

    private static Map<String, Object> assistantMessage(ChatCompletionResponseMessage responseMessage) {
        Map<String, Object> message = message("assistant", Optional.ofNullable(responseMessage.getContent()).orElse(""));
        if (responseMessage.getToolCalls() != null && !responseMessage.getToolCalls().isEmpty()) {
            message.put("tool_calls", JSON.convertValue(responseMessage.getToolCalls(), List.class));
        }
        return message;
    }

    private static Map<String, Object> toolMessage(String toolCallId, String content) {
        Map<String, Object> message = message("tool", content);
        message.put("tool_call_id", toolCallId);
        return message;
    }

    private static ObjectNode objectSchema() {
        ObjectNode node = JSON.createObjectNode();
        node.put("type", "object");
        return node;
    }

    private static ObjectNode schema(ObjectNode properties, ArrayNode required) {
        ObjectNode node = objectSchema();
        node.set("properties", properties);
        node.set("required", required);
        return node;
    }

    private static ObjectNode prop(String name, String type) {
        ObjectNode node = JSON.createObjectNode();
        node.put("name", name);
        node.put("type", type);
        return node;
    }

    private static ObjectNode props(ObjectNode... definitions) {
        ObjectNode props = JSON.createObjectNode();
        for (ObjectNode definition : definitions) {
            String name = definition.remove("name").asText();
            props.set(name, definition);
        }
        return props;
    }

    private static ArrayNode array(String... values) {
        ArrayNode array = JSON.createArrayNode();
        for (String value : values) {
            array.add(value);
        }
        return array;
    }

    private static String separator() {
        return DIM + "-".repeat(80) + RESET;
    }

    private static String preview(String value, int max) {
        Objects.requireNonNull(value, "value");
        String singleLine = value.replace('\n', ' ');
        return singleLine.length() <= max ? singleLine : singleLine.substring(0, max) + "...";
    }

    private String truncateToolResult(String result) {
        if (result.length() <= config.maxToolResultChars) {
            return result;
        }
        int omitted = result.length() - config.maxToolResultChars;
        return result.substring(0, config.maxToolResultChars)
                + "\n... (truncated, " + omitted + " chars omitted)";
    }

    private static void printContextSummary(List<Map<String, Object>> messages, long turnStartNanos) {
        int chars = messages.stream().mapToInt(message -> message.toString().length()).sum();
        long elapsedMillis = (System.nanoTime() - turnStartNanos) / 1_000_000L;
        System.out.println(DIM + "context: messages=" + messages.size()
                + " approx_chars=" + chars
                + " turn_ms=" + elapsedMillis + RESET);
    }

    public record Config(String baseUrl, String model, Integer ntokens, int maxTokens, int maxToolResultChars,
            int maxToolRounds, double temperature, boolean toolsEnabled, boolean allowRiskyTools, boolean streamEnabled,
            String javaSandboxImage, boolean enableThinking, boolean help) {
        static Config parse(String[] args) {
            if (args.length == 1 && ("--help".equals(args[0]) || "-h".equals(args[0]))) {
                return new Config("", "", null, 0, 0, 1, 0.0d, true, false, true, null, true, true);
            }
            if (args.length != 2 || !"--config".equals(args[0])) {
                throw new IllegalArgumentException("expected: --config <file>");
            }
            return fromJson(Path.of(args[1]));
        }

        public static Config fromJson(Path path) {
            try {
                ConfigFile file = JSON.readValue(path.toFile(), ConfigFile.class);
                if (file.maxToolRounds < 1) {
                    throw new IllegalArgumentException("maxToolRounds must be >= 1");
                }
                return new Config(stripTrailingSlash(file.baseUrl), file.model, file.ntokens, file.maxTokens,
                        file.maxToolResultChars, file.maxToolRounds, file.temperature, file.toolsEnabled,
                        file.allowRiskyTools, file.streamEnabled, file.javaSandboxImage, file.enableThinking, false);
            } catch (IOException e) {
                throw new IllegalArgumentException("could not read nanocode config " + path + ": " + e.getMessage(), e);
            }
        }

        private static String stripTrailingSlash(String value) {
            while (value.endsWith("/")) {
                value = value.substring(0, value.length() - 1);
            }
            return value;
        }

        static void printHelp() {
            System.out.println("""
                    nanocode-deliverance

                    Options:
                      --config FILE           JSON config file
                      --help                  Print this help

                    Config fields:
                      baseUrl, model, ntokens, maxTokens, maxToolResultChars,
                      maxToolRounds, temperature, toolsEnabled, allowRiskyTools,
                      streamEnabled, javaSandboxImage, enableThinking
                    """);
        }

        private static class ConfigFile {
            public String baseUrl = "http://localhost:8080";
            public String model = "default";
            public Integer ntokens;
            public int maxTokens = 2048;
            public int maxToolResultChars = 2000;
            public int maxToolRounds = 3;
            public double temperature = 0.0d;
            public boolean toolsEnabled = true;
            public boolean allowRiskyTools = false;
            public boolean streamEnabled = true;
            public String javaSandboxImage = "eclipse-temurin:25-jdk";
            public boolean enableThinking = true;
        }
    }

    private static final class SessionConfig {
        private int maxToolRounds;
        private boolean enableThinking;

        private SessionConfig(int maxToolRounds, boolean enableThinking) {
            this.maxToolRounds = maxToolRounds;
            this.enableThinking = enableThinking;
        }
    }
}
