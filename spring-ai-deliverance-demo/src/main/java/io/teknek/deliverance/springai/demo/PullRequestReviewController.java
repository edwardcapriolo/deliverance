package io.teknek.deliverance.springai.demo;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/reviews")
public class PullRequestReviewController {
    private final PullRequestReviewService reviewService;

    public PullRequestReviewController(PullRequestReviewService reviewService) {
        this.reviewService = reviewService;
    }

    @PostMapping("/pull-request")
    public PullRequestReviewResponse reviewPullRequest(@RequestBody PullRequestReviewRequest request) {
        return reviewService.reviewPullRequest(request);
    }

    @PostMapping("/release-note")
    public String releaseNote(@RequestBody PullRequestReviewRequest request) {
        return reviewService.releaseNote(request);
    }

    @PostMapping("/checklist")
    public String checklist(@RequestBody PullRequestReviewRequest request) {
        return reviewService.checklist(request);
    }

    @PostMapping("/classify")
    public String classify(@RequestBody PullRequestReviewRequest request) {
        return reviewService.classify(request);
    }
}
