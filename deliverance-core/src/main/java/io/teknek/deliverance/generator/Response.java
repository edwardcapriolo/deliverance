package io.teknek.deliverance.generator;

import io.teknek.deliverance.safetensors.prompt.ToolCall;

import java.util.Collections;
import java.util.List;

public class Response {
    public final String responseText;
    public final String responseTextWithSpecialTokens;
    public final FinishReason finishReason;
    public final int promptTokens;
    public final int generatedTokens;
    public final long promptTimeMs;
    public final long generateTimeMs;
    public final List<ToolCall> toolCalls;

    public Response(
            String responseText,
            String responseTextWithSpecialTokens,
            FinishReason finishReason,
            int promptTokens,
            int generatedTokens,
            long promptTimeMs,
            long generateTimeMs
    ) {
        this.responseText = responseText;
        this.responseTextWithSpecialTokens = responseTextWithSpecialTokens;
        this.finishReason = finishReason;
        this.promptTokens = promptTokens;
        this.generatedTokens = generatedTokens;
        this.promptTimeMs = promptTimeMs;
        this.generateTimeMs = generateTimeMs;
        this.toolCalls = Collections.emptyList();
    }

    private Response(
            String responseText,
            String responseTextWithSpecialTokens,
            FinishReason finishReason,
            int promptTokens,
            int generatedTokens,
            long promptTimeMs,
            long generateTimeMs,
            List<ToolCall> toolCalls
    ) {
        this.responseText = responseText;
        this.responseTextWithSpecialTokens = responseTextWithSpecialTokens;
        this.finishReason = finishReason;
        this.promptTokens = promptTokens;
        this.generatedTokens = generatedTokens;
        this.promptTimeMs = promptTimeMs;
        this.generateTimeMs = generateTimeMs;
        this.toolCalls = toolCalls;
    }

    public Response copyWithToolCalls(List<ToolCall> toolCalls) {
        return new Response(
                responseText,
                responseTextWithSpecialTokens,
                FinishReason.TOOL_CALL,
                promptTokens,
                generatedTokens,
                promptTimeMs,
                generateTimeMs,
                toolCalls
        );
    }

    @Override
    public String toString() {
        return "Response{"
                + "responseText='"
                + responseText
                + '\''
                + ", responseTextWithSpecialTokens='"
                + responseTextWithSpecialTokens
                + '\''
                + ", finishReason="
                + finishReason
                + ", promptTokens="
                + promptTokens
                + ", generatedTokens="
                + generatedTokens
                + ", promptTimeMs="
                + promptTimeMs
                + ", generateTimeMs="
                + generateTimeMs
                + '}';
    }
}