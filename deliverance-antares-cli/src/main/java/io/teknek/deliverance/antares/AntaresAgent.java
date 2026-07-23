package io.teknek.deliverance.antares;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class AntaresAgent {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final int NO_TOOL_RETRY_LIMIT = 5;
    private static final Pattern REPO_FILE_PATH = Pattern.compile("(?<![A-Za-z0-9_./-])(?:\\./)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+(?<![.,:;])");

    private final CompletionClient client;
    private final AntaresToolExecutor tools;
    private final int maxIterations;
    private final int maxToolCalls;
    private final int maxSubmitTurns;
    private final ToolCallParser parser = new ToolCallParser();

    AntaresAgent(CompletionClient client, AntaresToolExecutor tools, int maxIterations, int maxToolCalls) {
        this(client, tools, maxIterations, maxToolCalls, 3);
    }

    AntaresAgent(CompletionClient client, AntaresToolExecutor tools, int maxIterations, int maxToolCalls, int maxSubmitTurns) {
        this.client = client;
        this.tools = tools;
        this.maxIterations = maxIterations;
        this.maxToolCalls = maxToolCalls;
        this.maxSubmitTurns = maxSubmitTurns;
    }

    AgentResult run(String query) throws IOException {
        List<Message> messages = new ArrayList<>(AntaresPrompt.initialMessages(query, maxToolCalls));
        Set<String> seenToolCalls = new HashSet<>();
        LinkedHashSet<String> candidateFiles = new LinkedHashSet<>();
        StreamingConsoleRenderer renderer = new StreamingConsoleRenderer();
        System.err.println("[antares] repo root " + tools.repoRoot());
        String lastAssistantText = "";
        int noToolTurns = 0;
        int postBudgetSubmissionAttempts = 0;
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            boolean budgetExhaustedAtTurnStart = tools.toolCallsUsed() >= maxToolCalls;
            System.err.println("[antares] model turn " + (iteration + 1) + ", tool calls " + tools.toolCallsUsed()
                    + "/" + maxToolCalls);
            if (budgetExhaustedAtTurnStart) {
                System.err.println("[antares] submit-only turn " + (postBudgetSubmissionAttempts + 1) + "/" + maxSubmitTurns
                        + " after tool budget exhausted");
            }
            String assistantText = client.complete(messages, renderer::assistantChunk);
            renderer.endAssistantTurn();
            lastAssistantText = assistantText;
            captureCandidatePaths(candidateFiles, assistantText);
            messages.add(new Message("assistant", assistantText));
            List<ToolCall> calls = parser.parse(assistantText);
            if (calls.isEmpty()) {
                printNoToolDiagnostic(renderer, assistantText);
                if (budgetExhaustedAtTurnStart) {
                    postBudgetSubmissionAttempts++;
                    if (postBudgetSubmissionAttempts < maxSubmitTurns) {
                        messages.add(AntaresPrompt.submitOnlyRetry(maxToolCalls, List.copyOf(candidateFiles)));
                        continue;
                    }
                    return fallbackResult(candidateFiles, "Model did not emit submit_vulnerable_files after tool budget exhaustion. Last answer: "
                            + ToolCallParser.cleanAssistantText(lastAssistantText));
                }
                if (noToolTurns++ < NO_TOOL_RETRY_LIMIT) {
                    messages.add(AntaresPrompt.noToolRetry(noToolTurns - 1));
                    continue;
                }
                return fallbackResult(candidateFiles, "Model stopped without submitting. Last answer: "
                        + ToolCallParser.cleanAssistantText(lastAssistantText));
            }
            noToolTurns = 0;
            StringBuilder responses = new StringBuilder();
            boolean executedAny = false;
            for (ToolCall call : calls) {
                captureCandidatePaths(candidateFiles, call);
                String key = JSON.writeValueAsString(call);
                if (!seenToolCalls.add(key)) {
                    continue;
                }
                executedAny = true;
                renderer.toolCall(call);
                ToolResult result = tools.execute(call);
                ToolCall lastToolCall = tools.lastToolCall();
                if (lastToolCall != null) {
                    captureCandidatePaths(candidateFiles, lastToolCall);
                }
                if (result.submitted()) {
                    return result.result();
                }
                renderer.toolResult(result.response());
                captureCandidatePaths(candidateFiles, result.response());
                if (responses.length() > 0) {
                    responses.append("\n\n");
                }
                responses.append(result.response());
            }
            if (!executedAny) {
                messages.add(AntaresPrompt.duplicateRetry(false));
                continue;
            }
            int remaining = maxToolCalls - tools.toolCallsUsed();
            if (remaining <= 0) {
                if (budgetExhaustedAtTurnStart) {
                    postBudgetSubmissionAttempts++;
                }
                responses.append(AntaresPrompt.submissionRequiredNudge(maxToolCalls));
                if (!candidateFiles.isEmpty()) {
                    responses.append("\nObserved candidate files: ").append(candidateFiles);
                }
            } else {
                responses.append("\n[").append(remaining).append(" tool-calls remaining]");
            }
            messages.add(AntaresPrompt.toolResponse(responses.toString()));
            if (remaining <= 0 && postBudgetSubmissionAttempts >= maxSubmitTurns) {
                return fallbackResult(candidateFiles, "Model did not submit after repository tool budget was exhausted.");
            }
        }
        return fallbackResult(candidateFiles, "Maximum iterations reached. Last answer: "
                + ToolCallParser.cleanAssistantText(lastAssistantText));
    }

    private void printNoToolDiagnostic(StreamingConsoleRenderer renderer, String assistantText) {
        String cleaned = ToolCallParser.cleanAssistantText(assistantText);
        if (assistantText.toLowerCase().contains("<tool_call")) {
            renderer.assistantNotice("generated malformed or incomplete tool_call; no executable tool was parsed");
        } else if (cleaned.isBlank()) {
            renderer.assistantNotice("<empty/no parsed output>");
        }
    }

    private void captureCandidatePaths(Set<String> candidateFiles, ToolCall call) {
        if ("read_file".equals(call.name())) {
            captureCandidatePath(candidateFiles, firstString(call, "path", "file_path", "file"));
        }
        if ("terminal".equals(call.name()) || "bash".equals(call.name())) {
            captureCandidatePaths(candidateFiles, String.valueOf(call.arguments().getOrDefault("command", "")));
        }
        Object rankedFiles = call.arguments().get("ranked_files");
        if (rankedFiles instanceof List<?> list) {
            for (Object item : list) {
                captureCandidatePath(candidateFiles, item == null ? null : item.toString());
            }
        }
    }

    private void captureCandidatePaths(Set<String> candidateFiles, String text) {
        Matcher matcher = REPO_FILE_PATH.matcher(text);
        while (matcher.find()) {
            captureCandidatePath(candidateFiles, matcher.group());
        }
    }

    private void captureCandidatePath(Set<String> candidateFiles, String path) {
        if (path == null || path.isBlank()) {
            return;
        }
        String normalized = path.strip();
        if (normalized.startsWith("./")) {
            normalized = normalized.substring(2);
        }
        if (normalized.contains("..") || normalized.startsWith("/") || normalized.endsWith("/")) {
            return;
        }
        if (looksLikeSourceFile(normalized) && tools.submittedFileValidationError(normalized) == null) {
            candidateFiles.add(normalized);
        }
    }

    private boolean looksLikeSourceFile(String path) {
        return path.matches(".*\\.(java|kt|scala|py|js|ts|tsx|jsx|go|rs|c|cc|cpp|h|hpp|cs|rb|php|sql|xml|yaml|yml|json|properties)$");
    }

    private String firstString(ToolCall call, String... names) {
        for (String name : names) {
            Object value = call.arguments().get(name);
            if (value != null) {
                return value.toString();
            }
        }
        return null;
    }

    private AgentResult fallbackResult(LinkedHashSet<String> candidateFiles, String summary) {
        if (!candidateFiles.isEmpty()) {
            return AgentResult.fallbackVulnerable(List.copyOf(candidateFiles), summary);
        }
        return AgentResult.incomplete(summary);
    }
}
