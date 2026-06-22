package io.teknek.deliverance.safetensors.prompt.local;

import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Assumptions;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

final class LocalPromptTemplateFixtures {
    private LocalPromptTemplateFixtures() {
    }

    static PromptSupport promptSupport(String modelDirectory) {
        JsonUtils.om.findAndRegisterModules();
        Path tokenizerConfig = Path.of(System.getProperty("user.home"), ".deliverance", modelDirectory, "tokenizer_config.json");
        Assumptions.assumeTrue(Files.isRegularFile(tokenizerConfig), "Missing tokenizer config cache: " + tokenizerConfig);
        try {
            String template = JsonUtils.om.readTree(tokenizerConfig.toFile()).path("chat_template").asText(null);
            if (template == null) {
                Path chatTemplate = tokenizerConfig.getParent().resolve("chat_template.jinja");
                Assumptions.assumeTrue(Files.isRegularFile(chatTemplate), "Missing chat template cache: " + chatTemplate);
                template = Files.readString(chatTemplate);
            }
            String bos = JsonUtils.om.readTree(tokenizerConfig.toFile()).path("bos_token").asText("");
            String eos = JsonUtils.om.readTree(tokenizerConfig.toFile()).path("eos_token").asText("");
            return new PromptSupport(Map.of("default", template), bos, eos, template.contains("tool"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
