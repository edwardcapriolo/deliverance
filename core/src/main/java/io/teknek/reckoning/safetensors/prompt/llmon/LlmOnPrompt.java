package io.teknek.reckoning.safetensors.prompt.llmon;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.HashSet;
import java.util.Set;

import com.fasterxml.jackson.databind.JsonNode;

import io.teknek.reckoning.JsonUtils;

import static io.teknek.reckoning.safetensors.prompt.llmon.LlmOnSyntax.*;

/**
 * Builder and renderer for Granite LLMON prompts.
 *
 * <p>Use this class to compose a prompt from typed system messages, user instructions, data,
 * requirements, documents, tools, assistant tool calls, and tool results. The builder assigns
 * LLMON block IDs, creates execution bindings, flattens tool schemas, and serializes tool-call
 * arguments. Call {@link #render()} to obtain the complete LLMON prompt string to send to a
 * completion request, instead of assembling LLMON markup or JSON by hand.</p>
 *
 * <p>For example:</p>
 * <pre>{@code
 * String prompt = LlmOnPrompt.builder()
 *         .system("Be concise.")
 *         .user(LlmOnPrompt.UserMessage.builder()
 *                 .content("Summarize this document")
 *                 .document(new LlmOnPrompt.Document("Long document text"))
 *                 .build())
 *         .build()
 *         .render();
 * }</pre>
 *
 * <p>LLMON is described as an LLM-native markup language in the
 * <a href="https://arxiv.org/abs/2603.22519">LLMON language paper</a>. The
 * <a href="https://github.com/generative-computing/llmon">LLMON repository</a>
 * contains the reference renderer and examples. Granite-specific background is available in
 * the <a href="https://huggingface.co/blog/ibm-granite/llmon-an-llm-native-markup-language-now-in-granite">IBM Granite explainer</a>.</p>
 */
public final class LlmOnPrompt {

    private final List<Message> messages;
    private final boolean addGenerationPrompt;
    private final String eosToken;
    private final List<Document> documents;
    private final List<Tool> tools;

    private LlmOnPrompt(List<Message> messages, boolean addGenerationPrompt, String eosToken,
            List<Document> documents, List<Tool> tools) {
        this.messages = List.copyOf(messages);
        this.addGenerationPrompt = addGenerationPrompt;
        this.eosToken = Objects.requireNonNull(eosToken, "eosToken");
        this.documents = List.copyOf(documents);
        this.tools = List.copyOf(tools);
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Renders this prompt as LLMON text ready for the raw completions API. */
    public String render() {
        IdAllocator ids = new IdAllocator();
        StringBuilder output = new StringBuilder();
        String systemId = null;
        List<String> documentIds = new ArrayList<>();
        List<String> toolIds = new ArrayList<>();
        String executionBody = "";
        List<String> continuationInputs = new ArrayList<>();
        for (Document document : this.documents) {
            String id = ids.id(document.id());
            output.append(documentBlock(document, id));
            documentIds.add(id);
        }
        for (Tool tool : this.tools) {
            String id = ids.id(tool.id());
            output.append(toolBlock(tool, id));
            toolIds.add(id);
        }
        for (int index = 0; index < this.messages.size(); index++) {
            Message message = this.messages.get(index);
            if (message.role() == Role.SYSTEM) {
                systemId = ids.id(message.id());
                output.append(block(reference("system", systemId), message.content()));
            }
            else if (message.role() == Role.USER) {
                UserMessage user = message.user();
                String contentId = ids.id(user.contentId());
                String executionId = ids.id(user.executionId());
                String outputId = ids.id(user.outputId());
                List<String> instructions = new ArrayList<>();
                List<String> data = new ArrayList<>();
                List<String> requirements = new ArrayList<>();
                if (user.content() != null) {
                    output.append(block(reference("instr", contentId), user.content()));
                    instructions.add(reference("instr", contentId));
                }
                for (Document document : user.documents()) {
                    String id = ids.id(document.id());
                    output.append(documentBlock(document, id));
                    if (document.execute()) data.add(reference("doc", id));
                }
                for (Tool tool : user.tools()) {
                    String id = ids.id(tool.id());
                    output.append(toolBlock(tool, id));
                    if (tool.execute()) data.add(reference("tool", id));
                }
                for (Item item : user.instructions()) {
                    String id = ids.id(item.id());
                    output.append(block(reference("instr", id), item.value()));
                    if (item.execute()) instructions.add(reference("instr", id));
                }
                for (Item item : user.data()) {
                    String id = ids.id(item.id());
                    output.append(block(reference("data", id), item.value()));
                    if (item.execute()) data.add(reference("data", id));
                }
                for (Item item : user.requirements()) {
                    String id = ids.id(item.id());
                    output.append(block(reference("req", id), item.value()));
                    if (item.execute()) requirements.add(reference("req", id));
                }
                StringBuilder execution = new StringBuilder();
                execution.append(OPEN).append(reference("exec", executionId)).append(CLOSE);
                if (systemId != null) execution.append(pointer("system", reference("system", systemId)));
                for (String instruction : instructions) execution.append(pointer("instruction", instruction));
                for (String value : data) execution.append(pointer("input", value));
                for (String requirement : requirements) execution.append(pointer("requirement", requirement));
                for (String id : documentIds) execution.append(pointer("input", reference("doc", id)));
                for (String id : toolIds) execution.append(pointer("input", reference("tool", id)));
                execution.append(pointer("output", reference("assistant", outputId))).append(OPEN_END);
                output.append(execution);
                executionBody = execution.substring(0);
                continuationInputs.clear();
                if (this.addGenerationPrompt && index == this.messages.size() - 1) {
                    output.append(OPEN).append(reference("assistant", outputId)).append(CLOSE);
                }
            }
            else if (message.role() == Role.ASSISTANT) {
                String assistantId = ids.id(message.id());
                String assistantContent = message.assistant() == null
                        ? message.content() : assistantBody(message.assistant());
                output.append(block(reference("assistant", assistantId), assistantContent)).append(this.eosToken);
                if (message.assistant() != null && !message.assistant().toolCalls().isEmpty()) {
                    continuationInputs.add(reference("assistant", assistantId));
                }
            }
            if (message.role() == Role.TOOL) {
                String resultId = ids.id(message.id());
                output.append(block(reference("data", resultId),
                        TOOL_RESPONSE_OPEN + message.toolResult().content() + TOOL_RESPONSE_CLOSE));
                continuationInputs.add(reference("data", resultId));
                boolean lastTool = index == this.messages.size() - 1
                        || this.messages.get(index + 1).role() != Role.TOOL;
                if (lastTool) {
                    String executionId = ids.id(message.toolResult().executionId());
                    String outputId = ids.id(message.toolResult().outputId());
                    output.append(OPEN).append(reference("exec", executionId)).append(CLOSE)
                            .append(executionBody.substring(executionBody.indexOf(CLOSE) + CLOSE.length(),
                                    executionBody.length() - OPEN_END.length()));
                    for (String input : continuationInputs) output.append(pointer("input", input));
                    output.append(pointer("output", reference("assistant", outputId))).append(OPEN_END);
                    if (this.addGenerationPrompt && index == this.messages.size() - 1) {
                        output.append(OPEN).append(reference("assistant", outputId)).append(CLOSE);
                    }
                    continuationInputs.clear();
                }
            }
        }
        return output.toString();
    }

    private static String block(String tag, String value) {
        return OPEN + tag + CLOSE + (value == null ? "" : value) + OPEN_END;
    }

    private static String pointer(String role, String target) {
        return OPEN + "exec" + DOT + role + CLOSE + target + OPEN_END;
    }

    private static String documentBlock(Document document, String id) {
        String title = document.title() == null || document.title().isEmpty()
                ? OPEN + "doc" + DOT + "title" + SELF_CLOSE
                : block("doc" + DOT + "title", document.title());
        return block(reference("doc", id), title + block("doc" + DOT + "text", document.text()) + OPEN_END);
    }

    private static String toolBlock(Tool tool, String id) {
        StringBuilder result = new StringBuilder(block(reference("tool", id), ""));
        result.setLength(result.length() - OPEN_END.length());
        flatten(result, "tool", tool.schema());
        result.append(OPEN_END);
        return result.toString();
    }

    private static void flatten(StringBuilder output, String prefix, Object value) {
        if (value instanceof JsonNode node) value = JsonUtils.om.convertValue(node, Object.class);
        if (value instanceof Map<?, ?> map) {
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                String key = String.valueOf(entry.getKey());
                if ((prefix.equals("tool") || prefix.equals("tool" + DOT + "parameters")) && key.equals("type")) continue;
                String next = key.equals("function") || key.equals("properties") ? prefix : prefix + DOT + key;
                flatten(output, next, entry.getValue());
            }
        }
        else if (value instanceof Collection<?> collection) {
            for (Object item : collection) flatten(output, prefix, item);
        }
        else if (value != null) output.append(block(prefix, String.valueOf(value)));
    }

    private static String json(Object value) {
        if (value instanceof JsonNode node) return node.toString();
        if (value instanceof String string) {
            try { JsonUtils.om.readTree(string); return string; }
            catch (Exception ignored) { return JsonUtils.om.valueToTree(string).toString(); }
        }
        return JsonUtils.om.valueToTree(value).toString();
    }

    private static String assistantBody(AssistantMessage assistant) {
        StringBuilder result = new StringBuilder();
        if (assistant.content() != null && !assistant.content().isEmpty()) result.append(assistant.content());
        for (int index = 0; index < assistant.toolCalls().size(); index++) {
            if (result.length() > 0) result.append('\n');
            ToolCall call = assistant.toolCalls().get(index);
            result.append(TOOL_CALL_OPEN).append("{\"name\": ")
                    .append(JsonUtils.om.valueToTree(call.name()).toString())
                    .append(", \"arguments\": ").append(json(call.arguments()))
                    .append('}').append(TOOL_CALL_CLOSE);
        }
        return result.toString();
    }

    public static final class Builder {
        private final List<Message> messages = new ArrayList<>();
        private boolean addGenerationPrompt = true;
        private String eosToken = EOS;
        private final List<Document> documents = new ArrayList<>();
        private final List<Tool> tools = new ArrayList<>();

        public Builder system(String content) {
            this.messages.add(Message.system(content));
            return this;
        }

        public Builder user(UserMessage user) {
            this.messages.add(Message.user(Objects.requireNonNull(user, "user")));
            return this;
        }

        public Builder assistant(String content) {
            this.messages.add(Message.assistant(content));
            return this;
        }

        public Builder assistant(AssistantMessage assistant) {
            this.messages.add(Message.assistant(Objects.requireNonNull(assistant, "assistant")));
            return this;
        }

        public Builder toolResult(String content) {
            return toolResult(ToolResult.of(content));
        }

        public Builder toolResult(Object content) {
            return toolResult(ToolResult.of(json(content)));
        }

        public Builder toolResult(ToolResult result) {
            this.messages.add(Message.tool(Objects.requireNonNull(result, "result")));
            return this;
        }

        public Builder document(Document document) { this.documents.add(Objects.requireNonNull(document)); return this; }
        public Builder tool(Tool tool) { this.tools.add(Objects.requireNonNull(tool)); return this; }

        public Builder addGenerationPrompt(boolean addGenerationPrompt) {
            this.addGenerationPrompt = addGenerationPrompt;
            return this;
        }

        public Builder eosToken(String eosToken) {
            this.eosToken = Objects.requireNonNull(eosToken, "eosToken");
            return this;
        }

        public LlmOnPrompt build() {
            if (this.messages.isEmpty()) throw new IllegalArgumentException("At least one message is required");
            return new LlmOnPrompt(this.messages, this.addGenerationPrompt, this.eosToken, this.documents, this.tools);
        }
    }

    public static final class UserMessage {
        private final String content;
        private final List<Item> instructions;
        private final List<Item> data;
        private final List<Item> requirements;
        private final String contentId;
        private final String executionId;
        private final String outputId;
        private final List<Document> documents;
        private final List<Tool> tools;

        private UserMessage(UserBuilder builder) {
            this.content = builder.content;
            this.instructions = List.copyOf(builder.instructions);
            this.data = List.copyOf(builder.data);
            this.requirements = List.copyOf(builder.requirements);
            this.contentId = builder.contentId;
            this.executionId = builder.executionId;
            this.outputId = builder.outputId;
            this.documents = List.copyOf(builder.documents);
            this.tools = List.copyOf(builder.tools);
        }

        private String content() {
            return this.content;
        }

        private List<Item> instructions() {
            return this.instructions;
        }

        private List<Item> data() {
            return this.data;
        }

        private List<Item> requirements() {
            return this.requirements;
        }

        private String contentId() {
            return this.contentId;
        }

        private String executionId() {
            return this.executionId;
        }

        private String outputId() {
            return this.outputId;
        }

        private List<Document> documents() { return documents; }
        private List<Tool> tools() { return tools; }

        public static UserBuilder builder() {
            return new UserBuilder();
        }
    }

    public static final class UserBuilder {
        private final List<Item> instructions = new ArrayList<>();
        private final List<Item> data = new ArrayList<>();
        private final List<Item> requirements = new ArrayList<>();
        private String content;
        private String contentId;
        private String executionId;
        private String outputId;
        private final List<Document> documents = new ArrayList<>();
        private final List<Tool> tools = new ArrayList<>();

        public UserBuilder content(String content) {
            this.content = content;
            return this;
        }

        public UserBuilder instruction(String value) {
            return instruction(value, true);
        }

        public UserBuilder instruction(String value, boolean execute) {
            this.instructions.add(new Item(value, execute, null));
            return this;
        }

        public UserBuilder data(String value) {
            return data(value, true);
        }

        public UserBuilder data(String value, boolean execute) {
            this.data.add(new Item(value, execute, null));
            return this;
        }

        public UserBuilder data(String value, boolean execute, String id) {
            this.data.add(new Item(value, execute, id));
            return this;
        }

        public UserBuilder requirement(String value) {
            return requirement(value, true);
        }

        public UserBuilder requirement(String value, boolean execute) {
            this.requirements.add(new Item(value, execute, null));
            return this;
        }

        public UserBuilder contentId(String id) {
            this.contentId = id;
            return this;
        }

        public UserBuilder executionId(String id) {
            this.executionId = id;
            return this;
        }

        public UserBuilder outputId(String id) {
            this.outputId = id;
            return this;
        }

        public UserBuilder document(Document document) { documents.add(Objects.requireNonNull(document)); return this; }
        public UserBuilder tool(Tool tool) { tools.add(Objects.requireNonNull(tool)); return this; }

        public UserMessage build() {
            return new UserMessage(this);
        }
    }

    public record Item(String value, boolean execute, String id) {
        public Item {
            Objects.requireNonNull(value, "value");
        }
    }

    public record Document(String text, String title, boolean execute, String id) {
        public Document { Objects.requireNonNull(text, "text"); }
        public Document(String text) { this(text, null, true, null); }
        public Document(String text, String title) { this(text, title, true, null); }
    }

    public record Tool(Map<String, ?> schema, boolean execute, String id) {
        public Tool { schema = Map.copyOf(schema); }
        public Tool(Map<String, ?> schema) { this(schema, true, null); }
    }

    public record ToolCall(String name, Object arguments) {
        public ToolCall { Objects.requireNonNull(name, "name"); }
    }

    public record AssistantMessage(String content, List<ToolCall> toolCalls, String id) {
        public AssistantMessage { toolCalls = List.copyOf(toolCalls); }
        public AssistantMessage(String content, List<ToolCall> toolCalls) { this(content, toolCalls, null); }
        public static AssistantMessage toolCall(String name, Object arguments) {
            return new AssistantMessage(null, List.of(new ToolCall(name, arguments)));
        }
    }

    public record ToolResult(String content, String executionId, String outputId) {
        public ToolResult { Objects.requireNonNull(content, "content"); }
        public static ToolResult of(String content) { return new ToolResult(content, null, null); }
    }

    private enum Role { SYSTEM, USER, ASSISTANT, TOOL }

    private record Message(Role role, String content, UserMessage user, AssistantMessage assistant,
            ToolResult toolResult, String id) {
        private static Message system(String content) {
            return new Message(Role.SYSTEM, Objects.requireNonNull(content, "content"), null, null, null, null);
        }

        private static Message user(UserMessage user) {
            return new Message(Role.USER, null, user, null, null, null);
        }

        private static Message assistant(String content) {
            return new Message(Role.ASSISTANT, Objects.requireNonNull(content, "content"), null, null, null, null);
        }

        private static Message assistant(AssistantMessage assistant) {
            return new Message(Role.ASSISTANT, assistant.content(), null, assistant, null, assistant.id());
        }

        private static Message tool(ToolResult result) {
            return new Message(Role.TOOL, null, null, null, result, null);
        }
    }

    private static final class IdAllocator {
        private int next;
        private final Set<String> used = new HashSet<>();

        private String id(String pinned) {
            if (pinned != null) {
                validateId(pinned);
                if (!this.used.add(pinned)) {
                    throw new IllegalArgumentException("LLMON id must not be pinned more than once: " + pinned);
                }
                this.next += 2;
                return pinned;
            }
            int first = this.next++;
            int second = this.next++;
            String generated = "" + ID_CHARS.charAt(first % ID_CHARS.length())
                    + ID_CHARS.charAt(second % ID_CHARS.length());
            if (!this.used.add(generated)) {
                throw new IllegalStateException("LLMON generated duplicate ID: " + generated);
            }
            return generated;
        }

        private static void validateId(String id) {
            if (id.length() != 2 || id.chars().anyMatch(character -> ID_CHARS.indexOf(character) < 0)) {
                throw new IllegalArgumentException("LLMON id must contain two characters from the model ID alphabet");
            }
        }
    }

}
