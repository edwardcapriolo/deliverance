package io.teknek.deliverance.safetensors.prompt;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.hubspot.jinjava.Jinjava;
import com.hubspot.jinjava.JinjavaConfig;
import com.hubspot.jinjava.LegacyOverrides;
import com.hubspot.jinjava.interpret.RenderResult;
import com.hubspot.jinjava.lib.fn.ELFunctionDefinition;
import io.teknek.deliverance.JsonUtils;

import java.util.*;

import static io.teknek.deliverance.safetensors.fetch.HttpSupport.logger;

public class PromptSupport {

    private final Map<String, String> promptTemplates;
    private final String bosToken;
    private final String eosToken;
    private final boolean hasToolSupport;
    private final Jinjava jinjava;

    public PromptSupport(Map<String, String> promptTemplates, String eosToken, boolean hasToolSupport){
        this(promptTemplates, "", eosToken, hasToolSupport);
    }

    public PromptSupport(Map<String, String> promptTemplates, String bosToken, String eosToken, boolean hasToolSupport){
        this.promptTemplates = Map.copyOf(promptTemplates == null ? Map.of() : promptTemplates);
        this.bosToken = bosToken == null ? "" : bosToken;
        this.eosToken = eosToken == null ? "" : eosToken;
        this.hasToolSupport = hasToolSupport;
        jinjava = new Jinjava(
                JinjavaConfig.newBuilder()
                        .withTrimBlocks(true)
                        .withLstripBlocks(true)
                        .withLegacyOverrides(
                                LegacyOverrides.newBuilder()
                                        .withParseWhitespaceControlStrictly(true)
                                        .withUseTrimmingForNotesAndExpressions(true)
                                        .withUseSnakeCasePropertyNaming(true)
                                        .withKeepNullableLoopValues(true)
                                        .build()
                        )
                        .withObjectMapper(
                                new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT)
                                        .setDefaultPrettyPrinter(JsonUtils.JlamaPrettyPrinter.INSTANCE)
                        )
                        .build()
        );

        jinjava.getGlobalContext()
                .registerFunction(new ELFunctionDefinition("", "raise_exception", PromptSupport.class, "raiseException", String.class));
    }


    public static void raiseException(String message) {
        logger.error("Prompt template error: {}", message);
        throw new RuntimeException(message);
    }

    public Builder builder(){
        return new Builder(this.promptTemplates, this.bosToken, this.eosToken, this.hasToolSupport, this.jinjava);
    }

    public static class Builder {
        private final Map<String, String> promptTemplates;
        private final String bosToken;
        private final String eosToken;
        private final boolean hasToolSupport;
        private final Jinjava jinJava;
        private PromptType type = PromptType.DEFAULT;
        private boolean addGenerationPrompt = true;
        private String customizedTemplate;
        private final List<Tool> tools = new ArrayList<>();
        private final Map<String, Object> templateArgs = new HashMap<>();

        private final List<Message> messages = new ArrayList<>(2);

        private boolean stripPreamble = false;

        private Builder(Map<String, String> promptTemplates, String bosToken, String eosToken, boolean hasToolSupport, Jinjava jinJava) {
            this.promptTemplates = promptTemplates;
            this.bosToken = bosToken;
            this.eosToken = eosToken;
            this.hasToolSupport = hasToolSupport;
            this.jinJava = jinJava;
        }

        public Builder useChatTemplate(String templateString){
            customizedTemplate = templateString;
            return this;
        }

        public Builder usePromptType(PromptType type) {
            this.type = type;
            return this;
        }

        public Builder addGenerationPrompt(boolean addGenerationPrompt) {
            this.addGenerationPrompt = addGenerationPrompt;
            return this;
        }

        public Builder addTemplateArg(String name, Object value) {
            templateArgs.put(name, value);
            return this;
        }

        public Builder addTemplateArgs(Map<String, Object> values) {
            if (values != null) {
                templateArgs.putAll(values);
            }
            return this;
        }

        public Builder addUserMessage(String content) {
            messages.add(new Message(content, PromptRole.USER));
            return this;
        }

        public Builder addToolResult(ToolResult result) {
            messages.add(new Message(result));
            return this;
        }

        public Builder addToolCall(ToolCall call) {
            messages.add(new Message(call));
            return this;
        }

        public Builder addSystemMessage(String content) {
            messages.add(new Message(content, PromptRole.SYSTEM));
            return this;
        }

        public Builder addAssistantMessage(String content) {
            messages.add(new Message(content, PromptRole.ASSISTANT));
            return this;
        }

        public Builder stripPreamble() {
            stripPreamble = true;
            return this;
        }

        /* Adds a single tool call to the list */
        public Builder addToolItem(Tool tool){
            tools.add(tool);
            return this;
        }

        public PromptContext build() {
            return build(Optional.empty());
        }

        /**
         *
         * @param additionalTools adds the list of tools to the existing list inside the builder
         */
        public PromptContext build(List<Tool> additionalTools) {
            return build(Optional.of(additionalTools));
        }

        private PromptContext build(Optional<List<Tool>> optionalTools) {
            List<Tool> renderTools = new ArrayList<>(tools);
            optionalTools.ifPresent(renderTools::addAll);
            if (messages.isEmpty()) {
                throw new IllegalArgumentException("No messages to generate prompt");
            }
            if (promptTemplates.isEmpty()) {
                throw new UnsupportedOperationException("Prompt templates are not available for this model");
            }
            String template;
            if (customizedTemplate != null){
                template = customizedTemplate;
            } else {
                template = Optional.ofNullable(promptTemplates.get(type.name().toLowerCase()))
                        .orElseThrow(() -> new UnsupportedOperationException("Prompt template not available for type: " + type));
            }
            if (!renderTools.isEmpty() && !hasToolSupport) {
                throw new RuntimeException("This model does not support tools, but tools are specified");
            }
            if (template.contains("messages[::-1]") && template.contains("<think>")) {
                return new PromptContext(renderQwen3TemplateFallback(renderTools));
            }
            String preamble = "";
            if (stripPreamble) {
                Map<String, Object> args = new HashMap<>();
                args.putAll(Map.of("messages", Map.of(), "add_generation_prompt", false, "eos_token",
                        eosToken, "bos_token", bosToken));
                args.putAll(templateArgs);
                RenderResult r = jinJava.renderForResult(template, args);
                preamble = r.getOutput();
            }

            Map<String, Object> args = new HashMap<>();
            args.putAll(
                    Map.of( "messages", messages.stream().map(Message::toMap).toList(), "add_generation_prompt",
                            addGenerationPrompt, "eos_token", eosToken, "bos_token", bosToken)
            );
            args.putAll(templateArgs);
            if (!renderTools.isEmpty()){
                args.put("tools", renderTools);
            }
            RenderResult renderResult = jinJava.renderForResult(template, args);
            if (renderResult.hasErrors()) {
                logger.debug("Prompt template errors: {}", renderResult.getErrors());
                throw new RuntimeException("Prompt template errors: " + renderResult.getErrors());
            }
            String output = renderResult.getOutput();
            if (Boolean.TRUE.equals(templateArgs.get("enable_thinking")) && output.endsWith("<|turn>model\n")) {
                output = output + "<|channel>thought\n";
            }
            return new PromptContext(output.substring(preamble.length()));
        }

        private String renderQwen3TemplateFallback(List<Tool> renderTools) {
            StringBuilder output = new StringBuilder();
            List<Map> maps = messages.stream().map(Message::toMap).toList();
            if (!renderTools.isEmpty()) {
                output.append("<|im_start|>system\n");
                if (!maps.isEmpty() && "system".equals(maps.getFirst().get("role"))) {
                    output.append(maps.getFirst().get("content")).append("\n\n");
                }
                output.append("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n")
                        .append("You are provided with function signatures within <tools></tools> XML tags:\n<tools>");
                for (Tool tool : renderTools) {
                    try {
                        output.append('\n').append(JsonUtils.om.writeValueAsString(tool));
                    } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
                        throw new RuntimeException(e);
                    }
                }
                output.append("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n")
                        .append("<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n");
            } else if (!maps.isEmpty() && "system".equals(maps.getFirst().get("role"))) {
                output.append("<|im_start|>system\n").append(maps.getFirst().get("content")).append("<|im_end|>\n");
            }

            for (int i = 0; i < maps.size(); i++) {
                Map message = maps.get(i);
                String role = String.valueOf(message.get("role"));
                String content = String.valueOf(message.getOrDefault("content", ""));
                if ("system".equals(role) && i == 0) {
                    continue;
                }
                if ("user".equals(role) || "system".equals(role)) {
                    output.append("<|im_start|>").append(role).append('\n').append(content).append("<|im_end|>\n");
                    continue;
                }
                if ("assistant".equals(role)) {
                    output.append("<|im_start|>assistant\n").append(content == null ? "" : content);
                    Object toolCalls = message.get("tool_calls");
                    if (toolCalls instanceof List<?> list) {
                        for (Object tc : list) {
                            output.append("\n<tool_call>\n");
                            try {
                                output.append(JsonUtils.om.writeValueAsString(((Map<?, ?>) tc).get("function")));
                            } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
                                throw new RuntimeException(e);
                            }
                            output.append("\n</tool_call>");
                        }
                    }
                    output.append("<|im_end|>\n");
                    continue;
                }
                if ("tool".equals(role)) {
                    output.append("<|im_start|>user\n<tool_response>\n")
                            .append(content)
                            .append("\n</tool_response><|im_end|>\n");
                }
            }
            if (addGenerationPrompt) {
                output.append("<|im_start|>assistant\n");
                if (Boolean.FALSE.equals(templateArgs.get("enable_thinking"))) {
                    output.append("<think>\n\n</think>\n\n");
                }
            }
            return output.toString();
        }
    }

}
