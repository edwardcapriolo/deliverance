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
import io.teknek.deliverance.client.model.ChatCompletionRequestMessage;
import io.teknek.deliverance.client.model.ChatCompletionResponseMessage;
import io.teknek.deliverance.client.model.ChatCompletionTool;
import io.teknek.deliverance.client.model.CreateChatCompletionRequest;
import io.teknek.deliverance.client.model.CreateChatCompletionResponse;
import io.teknek.deliverance.client.model.CreateChatCompletionResponseChoicesInner;
import io.teknek.deliverance.client.model.FunctionObject;
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

    NanocodeDeliverance(Config config) {
        this.config = config;
        ApiClient apiClient = new ApiClient();
        apiClient.setAdapterBuilder(new Retrofit.Builder()
                .baseUrl(config.baseUrl + "/")
                .addConverterFactory(ScalarsConverterFactory.create())
                .addConverterFactory(JacksonConverterFactory.create(clientMapper())));
        apiClient.getOkBuilder().connectTimeout(Duration.ofSeconds(10));
        apiClient.getOkBuilder().readTimeout(Duration.ofMinutes(5));
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
                if (input.equals("/c")) {
                    messages = new ArrayList<>();
                    System.out.println(GREEN + "cleared" + RESET);
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

    private void runConversationTurn(List<Map<String, Object>> messages, String cwd) throws Exception {
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
            if (!content.isBlank()) {
                System.out.println(CYAN + "assistant" + RESET + " " + content);
            }
            messages.add(assistantMessage(responseMessage));

            List<ChatCompletionMessageToolCall> toolCalls = responseMessage.getToolCalls();
            if (toolCalls == null || toolCalls.isEmpty()) {
                return;
            }
            for (ChatCompletionMessageToolCall toolCall : toolCalls) {
                String id = toolCall.getId();
                String name = toolCall.getFunction().getName();
                JsonNode arguments = parseJsonObject(toolCall.getFunction().getArguments());
                System.out.println(GREEN + "tool " + name + RESET + " " + preview(arguments.toString(), 80));
                String result = truncateToolResult(runTool(name, arguments));
                System.out.println(DIM + preview(result, 120) + RESET);
                messages.add(toolMessage(id, result));
            }
        }
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private CreateChatCompletionResponse chat(List<Map<String, Object>> messages, String cwd) throws IOException {
        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model(config.model)
                .maxTokens(config.maxTokens)
                .temperature(BigDecimal.valueOf(config.temperature))
                .messages((List) withSystemMessage(messages, cwd))
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

    private List<Map<String, Object>> withSystemMessage(List<Map<String, Object>> messages, String cwd) {
        List<Map<String, Object>> result = new ArrayList<>();
        result.add(message("system", "You are a concise coding assistant. cwd: " + cwd
                + ". Use tools when needed. Prefer small, direct changes."));
        result.addAll(messages);
        return result;
    }

    List<ChatCompletionTool> toolSchema() {
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
        tools.add(tool("grep", "Search text files with a Java regular expression. Returns at most limit matches, default 10.", schema(
                props(prop("path", "string"), prop("pattern", "string"), prop("limit", "integer")),
                array("pattern"))));
        if (config.allowRiskyTools) {
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

    private String runTool(String name, JsonNode args) {
        try {
            return switch (name) {
                case "read" -> toolRead(args);
                case "write" -> toolWrite(args);
                case "edit" -> toolEdit(args);
                case "glob" -> toolGlob(args);
                case "grep" -> toolGrep(args);
                case "bash" -> config.allowRiskyTools ? toolBash(args) : "error: bash disabled; restart with --allow-risky-tools";
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

    static Map<String, Object> message(String role, String content) {
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

    record Config(String baseUrl, String model, Integer ntokens, int maxTokens, int maxToolResultChars,
            double temperature, boolean toolsEnabled, boolean allowRiskyTools, boolean help) {
        static Config parse(String[] args) {
            String baseUrl = env("DELIVERANCE_BASE_URL", "http://localhost:8080");
            String model = env("DELIVERANCE_MODEL", "default");
            Integer ntokens = optionalIntEnv("NANOCODE_NTOKENS");
            int maxTokens = Integer.parseInt(env("NANOCODE_MAX_TOKENS", "2048"));
            int maxToolResultChars = Integer.parseInt(env("NANOCODE_MAX_TOOL_RESULT_CHARS", "2000"));
            double temperature = Double.parseDouble(env("NANOCODE_TEMPERATURE", "0.0"));
            boolean toolsEnabled = Boolean.parseBoolean(env("NANOCODE_TOOLS", "true"));
            boolean allowRiskyTools = Boolean.parseBoolean(env("NANOCODE_ALLOW_RISKY_TOOLS", "false"));
            boolean help = false;
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--base-url" -> baseUrl = args[++i];
                    case "--model" -> model = args[++i];
                    case "--ntokens" -> ntokens = Integer.parseInt(args[++i]);
                    case "--max-tokens" -> maxTokens = Integer.parseInt(args[++i]);
                    case "--max-tool-result-chars" -> maxToolResultChars = Integer.parseInt(args[++i]);
                    case "--temperature" -> temperature = Double.parseDouble(args[++i]);
                    case "--no-tools" -> toolsEnabled = false;
                    case "--tools" -> toolsEnabled = true;
                    case "--allow-risky-tools" -> allowRiskyTools = true;
                    case "--help", "-h" -> help = true;
                    default -> throw new IllegalArgumentException("unknown argument: " + args[i]);
                }
            }
            return new Config(stripTrailingSlash(baseUrl), model, ntokens, maxTokens, maxToolResultChars,
                    temperature, toolsEnabled, allowRiskyTools, help);
        }

        private static String env(String name, String defaultValue) {
            return Optional.ofNullable(System.getenv(name)).orElse(defaultValue);
        }

        private static Integer optionalIntEnv(String name) {
            String value = System.getenv(name);
            return value == null || value.isBlank() ? null : Integer.parseInt(value);
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
                      --base-url URL          Deliverance base URL, default http://localhost:8080
                      --model MODEL           Model name, default DELIVERANCE_MODEL or default
                      --ntokens N             Optional total prompt+generation token budget; defaults to model context
                      --max-tokens N          Max response tokens, default 2048
                      --max-tool-result-chars N Max tool result chars kept in context, default 2000
                      --temperature VALUE     Temperature, default 0.0
                      --no-tools              Do not send tool definitions
                      --tools                 Send tool definitions
                      --allow-risky-tools     Enable risky/eval-prone bash tool
                      --help                  Print this help
                    """);
        }
    }
}
