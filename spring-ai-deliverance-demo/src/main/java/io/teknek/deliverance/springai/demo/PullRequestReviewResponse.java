package io.teknek.deliverance.springai.demo;

import java.util.List;

public record PullRequestReviewResponse(
        String summary,
        String riskLevel,
        List<Finding> findings,
        List<String> recommendedTests,
        String releaseNote) {

    public record Finding(String severity, String file, Integer line, String message) {
    }
}
