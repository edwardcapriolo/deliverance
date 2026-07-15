package io.teknek.deliverance.springai.demo;

import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.util.List;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest(properties = {
        "spring.ai.deliverance.mode=client",
        "spring.ai.deliverance.model=test-model"
})
@AutoConfigureMockMvc
class PullRequestReviewControllerTest {

    @Autowired
    MockMvc mockMvc;

    @Test
    void pullRequestReviewEndpointReturnsStructuredReview() throws Exception {
        mockMvc.perform(post("/api/reviews/pull-request")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "repository":"deliverance",
                                  "pullRequestId":"PR-1",
                                  "title":"Add guided JSON",
                                  "description":"Adds guided JSON support",
                                  "diff":"diff --git a/A.java b/A.java",
                                  "testOutput":"Tests run: 1, Failures: 0",
                                  "changedFiles":["A.java"]
                                }
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.summary").value("Adds guided decoding support."))
                .andExpect(jsonPath("$.riskLevel").value("low"));
    }

    @TestConfiguration
    static class TestConfig {
        @Bean
        ChatModel chatModel() {
            return new ChatModel() {
                @Override
                public ChatResponse call(Prompt prompt) {
                    return new ChatResponse(List.of(new Generation(new AssistantMessage("""
                            {"summary":"Adds guided decoding support.","riskLevel":"low","findings":[],"recommendedTests":["Run focused tests"],"releaseNote":"Adds guided decoding."}
                            """))));
                }
            };
        }
    }
}
