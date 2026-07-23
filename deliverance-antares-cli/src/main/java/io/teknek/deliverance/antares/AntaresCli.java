package io.teknek.deliverance.antares;

import com.fasterxml.jackson.databind.ObjectMapper;

public final class AntaresCli {
    private static final ObjectMapper JSON = new ObjectMapper();

    private AntaresCli() {
    }

    public static void main(String[] args) throws Exception {
        CliOptions options;
        try {
            options = CliOptions.parse(args);
        } catch (CliOptions.UsageException e) {
            System.err.println(e.getMessage());
            return;
        }
        OpenAiCompletionsClient client = new OpenAiCompletionsClient(options.endpoint, options.model,
                options.maxTokens, options.temperature, options.topP);
        ToolApproval approval = options.yesRunCommands ? ToolApproval.approveAll() : new ConsoleToolApproval();
        AntaresAgent agent = new AntaresAgent(client, new AntaresToolExecutor(options.repo, options.maxToolCalls,
                approval),
                options.maxIterations, options.maxToolCalls, options.maxSubmitTurns);
        AgentResult result = agent.run(CwePrompts.analysisPrompt(options.cwe, options.query));
        System.out.println(JSON.writerWithDefaultPrettyPrinter().writeValueAsString(result));
    }
}
