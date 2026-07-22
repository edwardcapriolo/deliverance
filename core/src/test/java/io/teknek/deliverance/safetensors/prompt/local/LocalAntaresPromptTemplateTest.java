package io.teknek.deliverance.safetensors.prompt.local;

import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.model.ReasoningTextSplitter;
import io.teknek.deliverance.safetensors.prompt.Function;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.safetensors.prompt.Tool;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class LocalAntaresPromptTemplateTest {

    @Test
    void rendersAntaresToolTemplateWhenCached() throws Exception {
        Path template = new ModelFetcher("fdtn-ai", "antares-1b-JQ4").pathForModel().resolve("chat_template.jinja");
        Assumptions.assumeTrue(Files.isRegularFile(template), "Antares JQ4 cache is not present: " + template);

        PromptSupport support = new PromptSupport(Map.of("default", Files.readString(template)), "<|end_of_text|>",
                "<|end_of_text|>", true);
        Tool grep = Tool.from(Function.builder()
                .name("grep")
                .description("Search files")
                .addParameter("pattern", Map.of("type", "string"), true)
                .build());

        PromptContext prompt = support.builder()
                .addToolItem(grep)
                .addUserMessage("Find CWE-78 candidates.")
                .build();

        assertTrue(prompt.getPrompt().contains("<tools>"), prompt.getPrompt());
        assertTrue(prompt.getPrompt().contains("grep"), prompt.getPrompt());
        assertTrue(prompt.getPrompt().contains("<|start_of_role|>assistant<|end_of_role|><think>"), prompt.getPrompt());
    }

    @Test
    void rendersNanocodeSecurityShowcasePromptWhenCached() throws Exception {
        Path template = new ModelFetcher("fdtn-ai", "antares-1b-JQ4").pathForModel().resolve("chat_template.jinja");
        Assumptions.assumeTrue(Files.isRegularFile(template), "Antares JQ4 cache is not present: " + template);

        PromptSupport support = new PromptSupport(Map.of("default", Files.readString(template)), "<|end_of_text|>",
                "<|end_of_text|>", true);

        PromptContext prompt = support.builder()
                .addSystemMessage("You are Antares running inside nanocode-deliverance. You are a vulnerability "
                        + "localization agent for defensive security work. cwd: /workspace/repo.")
                .addToolItem(tool("read", "Read a text file with line numbers.",
                        parameter("path", "string", true),
                        parameter("offset", "integer", false),
                        parameter("limit", "integer", false)))
                .addToolItem(tool("glob", "Find files by glob pattern.",
                        parameter("path", "string", false),
                        parameter("pattern", "string", true)))
                .addToolItem(tool("grep", "Search text files with a Java regular expression.",
                        parameter("path", "string", false),
                        parameter("pattern", "string", true),
                        parameter("limit", "integer", false)))
                .addToolItem(tool("bash", "Risky/eval tool. Run a shell command with a timeout.",
                        parameter("command", "string", true)))
                .addUserMessage("""
                        Security localization task:

                        Repository: ./showcase-security-repo

                        CWE-78: Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection'). The software constructs all or part of an OS command using externally influenced input, but does not neutralize special elements that could modify the intended command.

                        Use the available tools to inspect the repository. Return the file path or paths most likely to contain the vulnerability and cite the specific evidence you found. Do not provide exploit instructions.
                        """)
                .build();

        String rendered = prompt.getPrompt();
        assertTrue(rendered.contains("<tools>"), rendered);
        assertTrue(rendered.contains("\"name\": " + "\"grep\""), rendered);
        assertTrue(rendered.contains("\"name\": " + "\"bash\""), rendered);
        assertTrue(rendered.contains("CWE-78"), rendered);
        assertTrue(rendered.contains("./showcase-security-repo"), rendered);
        assertTrue(rendered.contains("<|start_of_role|>assistant<|end_of_role|><think>"), rendered);
        assertTrue(ReasoningTextSplitter.promptEndsInsideReasoning(rendered),
                "Antares generation starts inside reasoning because the template opens <think> in the prompt");
    }

    private static Tool tool(String name, String description, Parameter... parameters) {
        Function.Builder builder = Function.builder().name(name).description(description);
        for (Parameter parameter : parameters) {
            builder.addParameter(parameter.name(), Map.of("type", parameter.type()), parameter.required());
        }
        return Tool.from(builder.build());
    }

    private static Parameter parameter(String name, String type, boolean required) {
        return new Parameter(name, type, required);
    }

    private record Parameter(String name, String type, boolean required) {
    }
}
