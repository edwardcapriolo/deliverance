package io.teknek.deliverance.safetensors.prompt;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.hubspot.jinjava.Jinjava;
import com.hubspot.jinjava.JinjavaConfig;
import com.hubspot.jinjava.LegacyOverrides;
import com.hubspot.jinjava.interpret.RenderResult;
import com.hubspot.jinjava.lib.fn.ELFunctionDefinition;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.tokenizer.TokenizerModel;

import java.util.*;

import static io.teknek.deliverance.safetensors.fetch.HttpSupport.logger;

public class PromptSupport {

    private final TokenizerModel tokenizerModel;
    private final Jinjava jinjava;

    public PromptSupport(TokenizerModel tokenizerModel){
        this.tokenizerModel = tokenizerModel;
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
        return new Builder(this.tokenizerModel, this.jinjava);
    }

    public static class Builder {
        private final TokenizerModel m;
        private final Jinjava jinJava;
        private PromptType type = PromptType.DEFAULT;
        private boolean addGenerationPrompt = true;

        private final List<Message> messages = new ArrayList<>(2);
        private boolean stripPreamble = false;

        private Builder(TokenizerModel m, Jinjava jinJava) {
            this.m = m;
            this.jinJava = jinJava;
        }

        public Builder usePromptType(PromptType type) {
            this.type = type;
            return this;
        }

        public Builder addGenerationPrompt(boolean addGenerationPrompt) {
            this.addGenerationPrompt = addGenerationPrompt;
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

        public PromptContext build() {
            return build(Optional.empty());
        }


        public PromptContext build(List<Tool> tools) {
            return build(Optional.of(tools));
        }

        public PromptContext build(Tool... tools) {
            return build(Optional.of(List.of(tools)));
        }

        private PromptContext build(Optional<List<Tool>> optionalTools) {
            if (messages.isEmpty()) {
                throw new IllegalArgumentException("No messages to generate prompt");
            }
            if (m.getPromptTemplates().isEmpty()) {
                throw new UnsupportedOperationException("Prompt templates are not available for this model");
            }
            String template = m.getPromptTemplates()
                    .map(t -> t.get(type.name().toLowerCase()))
                    .orElseThrow(() -> new UnsupportedOperationException("Prompt template not available for type: " + type));
            if (optionalTools.isPresent() && !optionalTools.get().isEmpty() && !m.hasToolSupport()) {
                logger.warn(
                        "This model does not support tools, but tools are specified"
                );
            }
            String preamble = "";
            if (stripPreamble) {
                Map<String, Object> args = new HashMap<>();
                args.putAll(Map.of("messages", Map.of(), "add_generation_prompt", false, "eos_token",
                        m.getEosToken(), "bos_token", ""));
                RenderResult r = jinJava.renderForResult(template, args);
                preamble = r.getOutput();
            }

            Map<String, Object> args = new HashMap<>();
            args.putAll(
                    Map.of(
                            "messages",
                            messages.stream().map(Message::toMap).toList(),
                            "add_generation_prompt",
                            addGenerationPrompt,
                            "eos_token",
                            m.getEosToken(),
                            "bos_token",
                            ""
                    )
            );
            optionalTools.ifPresent(tools -> args.put("tools", tools));
            RenderResult r = jinJava.renderForResult(template, args);
            if (r.hasErrors()) {
                logger.debug("Prompt template errors: {}" ,r.getErrors());
                throw new RuntimeException("Prompt template errors: " + r.getErrors());
            }
            String output = r.getOutput();
            return new PromptContext(output.substring(preamble.length()), optionalTools);
        }
    }

}
