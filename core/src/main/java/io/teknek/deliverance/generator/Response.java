package io.teknek.deliverance.generator;

import io.teknek.deliverance.model.SamplerReturn;
import io.teknek.deliverance.safetensors.prompt.ToolCall;

import java.util.Collections;
import java.util.List;

public class  Response {
    public final String responseText;
    public final String responseTextWithSpecialTokens;
    public final String reasoning;
    public final FinishReason finishReason;
    public final int promptTokens;
    public final List<Integer> generatedTokens;
    public final List<SamplerReturn> samplerReturns;
    public final long promptTimeMs;
    public final long generateTimeMs;
    public final double timeToFirstTokenMs;
    public final double avgTimePerTokenMs;
    public final double totalTimeMs;
    public final List<ToolCall> toolCalls;

    public Response(
            String responseText,
            String responseTextWithSpecialTokens,
            FinishReason finishReason,
            int promptTokens,
            List<Integer> generatedTokens,
            long promptTimeMs,
            long generateTimeMs,
            List<SamplerReturn> samplerReturns
    ) {
        this(
                responseText,
                responseTextWithSpecialTokens,
                null,
                finishReason,
                promptTokens,
                generatedTokens,
                promptTimeMs,
                generateTimeMs,
                0.0,
                0.0,
                0.0,
                Collections.emptyList(),
                samplerReturns
        );
    }

    private Response(
            String responseText,
            String responseTextWithSpecialTokens,
            String reasoning,
            FinishReason finishReason,
            int promptTokens,
            List<Integer> generatedTokens,
            long promptTimeMs,
            long generateTimeMs,
            double timeToFirstTokenMs,
            double avgTimePerTokenMs,
            double totalTimeMs,
            List<ToolCall> toolCalls,
            List<SamplerReturn> samplerReturns
    ) {
        this.responseText = responseText;
        this.responseTextWithSpecialTokens = responseTextWithSpecialTokens;
        this.reasoning = reasoning;
        this.finishReason = finishReason;
        this.promptTokens = promptTokens;
        this.generatedTokens = generatedTokens;
        this.promptTimeMs = promptTimeMs;
        this.generateTimeMs = generateTimeMs;
        this.timeToFirstTokenMs = timeToFirstTokenMs;
        this.avgTimePerTokenMs = avgTimePerTokenMs;
        this.totalTimeMs = totalTimeMs;
        this.toolCalls = toolCalls;
        this.samplerReturns = samplerReturns;
    }

    public Response copyWithToolCalls(List<ToolCall> toolCalls) {
        return new Response(
                responseText,
                responseTextWithSpecialTokens,
                reasoning,
                FinishReason.TOOL_CALLS,
                promptTokens,
                generatedTokens,
                promptTimeMs,
                generateTimeMs,
                timeToFirstTokenMs,
                avgTimePerTokenMs,
                totalTimeMs,
                toolCalls,
                samplerReturns
        );
    }

    public Response copyWithText(String responseText, String responseTextWithSpecialTokens, String reasoning) {
        return new Response(
                responseText,
                responseTextWithSpecialTokens,
                reasoning,
                finishReason,
                promptTokens,
                generatedTokens,
                promptTimeMs,
                generateTimeMs,
                timeToFirstTokenMs,
                avgTimePerTokenMs,
                totalTimeMs,
                toolCalls,
                samplerReturns
        );
    }

    public Response copyWithTiming(double timeToFirstTokenMs, double avgTimePerTokenMs, double totalTimeMs) {
        return new Response(
                responseText,
                responseTextWithSpecialTokens,
                reasoning,
                finishReason,
                promptTokens,
                generatedTokens,
                Math.round(timeToFirstTokenMs),
                Math.round(totalTimeMs),
                timeToFirstTokenMs,
                avgTimePerTokenMs,
                totalTimeMs,
                toolCalls,
                samplerReturns
        );
    }

    @Override
    public String toString() {
        return "Response{" +
                "responseText='" + responseText + '\'' +
                ", responseTextWithSpecialTokens='" + responseTextWithSpecialTokens + '\'' +
                ", reasoning='" + reasoning + '\'' +
                ", finishReason=" + finishReason +
                ", promptTokens=" + promptTokens +
                ", generatedTokens=" + generatedTokens +
                ", samplerReturns=" + samplerReturns +
                ", promptTimeMs=" + promptTimeMs +
                ", generateTimeMs=" + generateTimeMs +
                ", timeToFirstTokenMs=" + timeToFirstTokenMs +
                ", avgTimePerTokenMs=" + avgTimePerTokenMs +
                ", totalTimeMs=" + totalTimeMs +
                ", toolCalls=" + toolCalls +
                '}';
    }
}
