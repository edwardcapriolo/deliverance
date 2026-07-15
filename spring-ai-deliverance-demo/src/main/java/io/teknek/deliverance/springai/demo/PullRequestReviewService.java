package io.teknek.deliverance.springai.demo;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.teknek.deliverance.springai.DeliveranceChatOptions;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;

@Service
public class PullRequestReviewService {
    private static final String REVIEW_SCHEMA = """
            {
              "type": "object",
              "properties": {
                "summary": { "type": "string" },
                "riskLevel": { "enum": ["low", "medium", "high"] },
                "findings": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      "severity": { "enum": ["low", "medium", "high"] },
                      "file": { "type": "string" },
                      "line": { "type": "integer" },
                      "message": { "type": "string" }
                    },
                    "required": ["severity", "file", "line", "message"],
                    "additionalProperties": false
                  }
                },
                "recommendedTests": { "type": "array", "items": { "type": "string" } },
                "releaseNote": { "type": "string" }
              },
              "required": ["summary", "riskLevel", "findings", "recommendedTests", "releaseNote"],
              "additionalProperties": false
            }
            """;

    private final ChatModel chatModel;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public PullRequestReviewService(ChatModel chatModel) {
        this.chatModel = chatModel;
    }

    public PullRequestReviewResponse reviewPullRequest(PullRequestReviewRequest request) {
        String response = chatModel.call(new Prompt(render("pr-review.st", request), DeliveranceChatOptions.builder()
                .temperature(0.0)
                .maxTokens(512)
                .guidedJson(REVIEW_SCHEMA)
                .build())).getResult().getOutput().getText();
        try {
            return objectMapper.readValue(response, PullRequestReviewResponse.class);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public String releaseNote(PullRequestReviewRequest request) {
        return chatModel.call(new Prompt(render("release-note.st", request))).getResult().getOutput().getText();
    }

    public String checklist(PullRequestReviewRequest request) {
        return chatModel.call(new Prompt(render("checklist.st", request))).getResult().getOutput().getText();
    }

    public String classify(PullRequestReviewRequest request) {
        return chatModel.call(new Prompt(render("classify.st", request), DeliveranceChatOptions.builder()
                .temperature(0.0)
                .guidedChoice(java.util.List.of("feature", "bugfix", "performance", "docs", "internal"))
                .build())).getResult().getOutput().getText();
    }

    private String render(String template, PullRequestReviewRequest request) {
        try {
            String value = new String(getClass().getResourceAsStream("/prompts/" + template).readAllBytes(),
                    StandardCharsets.UTF_8);
            return value
                    .replace("{{repository}}", nullToEmpty(request.repository()))
                    .replace("{{pullRequestId}}", nullToEmpty(request.pullRequestId()))
                    .replace("{{title}}", nullToEmpty(request.title()))
                    .replace("{{description}}", nullToEmpty(request.description()))
                    .replace("{{sourceBranch}}", nullToEmpty(request.sourceBranch()))
                    .replace("{{targetBranch}}", nullToEmpty(request.targetBranch()))
                    .replace("{{changedFiles}}", request.changedFiles() == null ? "" : String.join("\n", request.changedFiles()))
                    .replace("{{diff}}", nullToEmpty(request.diff()))
                    .replace("{{testOutput}}", nullToEmpty(request.testOutput()));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private String nullToEmpty(String value) {
        return value == null ? "" : value;
    }
}
