package io.teknek.deliverance.springai.demo;

import java.util.List;

public record PullRequestReviewRequest(
        String repository,
        String pullRequestId,
        String title,
        String description,
        String sourceBranch,
        String targetBranch,
        String diff,
        String testOutput,
        List<String> changedFiles) {
}
