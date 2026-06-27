package net.deliverance.http;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.*;
import io.teknek.deliverance.model.Error;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.safetensors.prompt.ToolCall;
import io.teknek.dysfx.Either;
import io.teknek.dysfx.Left;
import io.teknek.dysfx.Right;
import io.teknek.dysfx.exception.UnreachableException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.*;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;


@RestController
public class ChatCompletionController {
    private static final Logger LOGGER = LoggerFactory.getLogger(ChatCompletionController.class);

    private static final String DELIVERANCE_SESSION_HEADER = "X-Deliverance-Session";

    @Autowired
    @Qualifier("causalLanguageModels")
    private Map<MultiModelConfig, CausalLanguageModel> models;

    private Optional<Map.Entry<MultiModelConfig, CausalLanguageModel>> findModel(String name){
        return models.entrySet().stream()
                .filter(x-> x.getKey().getModelName()
                        .equalsIgnoreCase(name)).findFirst();
    }

    private PreGenerateSlot slot;
    private final boolean debugChatRequest;

    public ChatCompletionController(Optional<PreGenerateSlot> slot,
            @Value("${deliverance.debug.chat-request:false}") boolean debugChatRequest,
            @Value("${debug:false}") boolean springDebug){
        this.slot = slot.orElse((x,y) -> Either.Right(y));
        this.debugChatRequest = debugChatRequest || springDebug;
    }

    @RequestMapping(method = RequestMethod.POST, value = "/chat/completions", produces = { "application/json",
            "text/event-stream" }, consumes = { "application/json" })
    Object createChatCompletion(@RequestHeader Map<String, String> headers,
                                @RequestBody CreateChatCompletionRequest request) {
        Optional<Map.Entry<MultiModelConfig, CausalLanguageModel>> z = findModel(request.getModel());
        if (z.isEmpty()){
            return badRequest("model not found " + request.getModel());
        }
        CausalLanguageModel model = z.get().getValue();
        debugRequest(request);

        if (request.getStream() == null || (request.getStream() != null && request.getStream() == false)) {
            try {
                long mapStart = System.nanoTime();
                Either<Error, PreparedRequest> bla = ChatCompletionService.mapRequest(headers, model, request);
                debugElapsed("chat.map_request", mapStart);
                if (bla.isLeft()) {
                    Left<Error, ?> l = (Left<Error, ?>) bla;
                    Error r = (Error) l.productIterator().next();
                    return new ResponseEntity<>(new ErrorResponse().error(r), HttpStatus.BAD_REQUEST);
                } else {
                    PreparedRequest ready = (PreparedRequest) bla.productIterator().next();
                    Either<Error, PreparedRequest> afterSlot = slot.handle(request, ready);
                    switch (afterSlot) {
                        case Left<Error, PreparedRequest> e -> {
                            return new ResponseEntity<>(new ErrorResponse().error(e.get()), HttpStatus.BAD_REQUEST);
                        }
                        case Right<Error, PreparedRequest> r -> {
                            ready = r.get();
                        }
                    }
                    var promptContext = ready.promptSupportBuilder().build();
                    debugPrompt(promptContext.getPrompt());
                    long generateStart = System.nanoTime();
                    Response resp = model.generate(UUID.randomUUID(), promptContext,
                            ready.generatorParameters(), new DoNothingGenerateEvent());
                    debugElapsed("chat.generate", generateStart);
                    CreateChatCompletionResponse response = new CreateChatCompletionResponse();
                    List<ToolCall> tcs = model.getToolCallParser().extract(resp);
                    ChatCompletionResponseMessage message = new ChatCompletionResponseMessage()
                            .role(ChatCompletionResponseMessage.RoleEnum.ASSISTANT);
                    if (resp.reasoning != null) {
                        message.reasoning(resp.reasoning);
                    }
                    CreateChatCompletionResponseChoicesInner choice = new CreateChatCompletionResponseChoicesInner()
                            .message(message)
                            .index(0);
                    CreateChatCompletionResponseChoicesInnerLogprobs logprobs = toLogProbs(resp);
                    if (logprobs != null) {
                        choice.logprobs(logprobs);
                    }
                    if (!tcs.isEmpty()) {
                        message.content(null);
                        //We shouldn't need this but it seems like the behaivor is hadto model
                        choice.setFinishReason(CreateChatCompletionResponseChoicesInner.FinishReasonEnum.TOOL_CALLS);
                    } else {
                        message.content(resp.responseText);
                        switch (resp.finishReason){
                            case MAX_TOKENS -> choice.setFinishReason(CreateChatCompletionResponseChoicesInner.FinishReasonEnum.LENGTH);
                            case TOOL_CALLS -> choice.setFinishReason(CreateChatCompletionResponseChoicesInner.FinishReasonEnum.TOOL_CALLS);
                            case STOP_TOKEN -> choice.setFinishReason(CreateChatCompletionResponseChoicesInner.FinishReasonEnum.STOP);
                            case CONTENT_FILTER, FUNCTION_CALL -> throw new UnreachableException("not implemented");
                            case null -> throw new UnreachableException("unexpected null return");
                        }
                    }
                    tcs.forEach(tc -> {
                        ChatCompletionMessageToolCall t = new ChatCompletionMessageToolCall();
                        t.id(tc.getId());
                        t.function(new ChatCompletionMessageToolCallFunction().name(tc.getName()));
                        try {
                            String paramsAsString = JsonUtils.om.writeValueAsString(tc.getParameters());
                            t.getFunction().arguments(paramsAsString);
                        } catch (JsonProcessingException e) {
                            throw new RuntimeException(e);
                        }
                        choice.getMessage().addToolCallsItem(t);
                    });
                    response.addChoicesItem(choice);
                    return new ResponseEntity<>(response, HttpStatus.OK);
                }
            } catch (IllegalArgumentException | GenerationException e) {
                return badRequest(e.getMessage());
            }
        }

        return streamChatCompletion(headers, request, model);
    }

    private Object streamChatCompletion(Map<String, String> headers, CreateChatCompletionRequest request,
            CausalLanguageModel model) {
        UUID sessionId = UUID.randomUUID();
        if (headers.containsKey(DELIVERANCE_SESSION_HEADER)) {
            try {
                sessionId = UUID.fromString(headers.get(DELIVERANCE_SESSION_HEADER));
            } catch (IllegalArgumentException e) {
                return badRequest("invalid " + DELIVERANCE_SESSION_HEADER);
            }
        }
        Either<Error, PreparedRequest> mapped = ChatCompletionService.mapRequest(headers, model, request);
        if (mapped.isLeft()) {
            Left<Error, ?> left = (Left<Error, ?>) mapped;
            return new ResponseEntity<>(new ErrorResponse().error((Error) left.productIterator().next()), HttpStatus.BAD_REQUEST);
        }
        PreparedRequest ready = (PreparedRequest) mapped.productIterator().next();
        Either<Error, PreparedRequest> afterSlot = slot.handle(request, ready);
        switch (afterSlot) {
            case Left<Error, PreparedRequest> e -> {
                return new ResponseEntity<>(new ErrorResponse().error(e.get()), HttpStatus.BAD_REQUEST);
            }
            case Right<Error, PreparedRequest> r -> ready = r.get();
        }

        SseEmitter emitter = newSseEmitter();
        AtomicBoolean closed = new AtomicBoolean(false);
        emitter.onCompletion(() -> closed.set(true));
        emitter.onTimeout(() -> closed.set(true));
        emitter.onError(ignored -> closed.set(true));
        UUID finalSessionId = sessionId;
        PreparedRequest finalReady = ready;
        boolean toolsPresent = request.getTools() != null && !request.getTools().isEmpty();
        CompletableFuture.runAsync(() -> {
            try {
                var promptContext = finalReady.promptSupportBuilder().build();
                debugPrompt(promptContext.getPrompt());
                StringBuilder streamedText = new StringBuilder();
                StringBuilder pendingContent = new StringBuilder();
                AtomicBoolean suppressContent = new AtomicBoolean(false);
                Response response = model.generate(finalSessionId, promptContext, finalReady.generatorParameters(),
                        (int next, String tok, String token, float f) -> {
                            streamedText.append(token == null ? "" : token);
                            if (toolsPresent && !suppressContent.get()) {
                                pendingContent.append(token == null ? "" : token);
                                String flush = streamableToolAwareContent(pendingContent);
                                if (!flush.isEmpty()) {
                                    sendStreamToken(emitter, closed, finalSessionId, flush);
                                    throwIfStreamClosed(closed);
                                }
                            }
                            if (toolsPresent && (streamedText.indexOf("<tools>") >= 0
                                    || streamedText.indexOf("<tool_call>") >= 0)) {
                                suppressContent.set(true);
                                pendingContent.setLength(0);
                            }
                            if (!toolsPresent && !suppressContent.get()) {
                                sendStreamToken(emitter, closed, finalSessionId, token);
                                throwIfStreamClosed(closed);
                            }
                        });
                if (!closed.get()) {
                    List<ToolCall> toolCalls = extractToolCalls(model, response, streamedText.toString());
                    if (!toolCalls.isEmpty()) {
                        for (int i = 0; i < toolCalls.size(); i++) {
                            sendStreamEvent(emitter, closed, streamToolCallDelta(finalSessionId, i, toolCalls.get(i)));
                        }
                        sendStreamEvent(emitter, closed, streamComplete(finalSessionId, io.teknek.deliverance.generator.FinishReason.TOOL_CALLS));
                    } else {
                        if (toolsPresent && pendingContent.length() > 0) {
                            sendStreamToken(emitter, closed, finalSessionId, pendingContent.toString());
                        }
                        sendStreamEvent(emitter, closed, streamComplete(finalSessionId, response.finishReason));
                    }
                    if (!closed.get()) {
                        sendStreamEvent(emitter, closed, SseEmitter.event().data("[DONE]"));
                    }
                    if (!closed.get()) {
                        closed.set(true);
                        emitter.complete();
                    }
                }
            } catch (Throwable t) {
                if (!closed.get()) {
                    closed.set(true);
                    emitter.completeWithError(t);
                }
            }
        });
        return emitter;
    }

    protected SseEmitter newSseEmitter() {
        return new SseEmitter(-1L);
    }

    private static void throwIfStreamClosed(AtomicBoolean closed) {
        if (closed.get()) {
            throw new CancellationException("stream closed");
        }
    }

    private static List<ToolCall> extractToolCalls(CausalLanguageModel model, Response response, String streamedText) {
        List<ToolCall> toolCalls = model.getToolCallParser().extract(response);
        if (!toolCalls.isEmpty() || streamedText == null || streamedText.isBlank()
                || streamedText.equals(response.responseTextWithSpecialTokens)) {
            return toolCalls;
        }
        Response streamedResponse = new Response(response.responseText, streamedText, response.finishReason,
                response.promptTokens, response.generatedTokens, response.promptTimeMs, response.generateTimeMs,
                response.samplerReturns);
        return model.getToolCallParser().extract(streamedResponse);
    }

    private static String streamableToolAwareContent(StringBuilder pendingContent) {
        int tagStart = firstToolTagIndex(pendingContent);
        if (tagStart >= 0) {
            String flush = pendingContent.substring(0, tagStart);
            pendingContent.setLength(0);
            return flush;
        }
        int keep = longestToolTagPrefixSuffix(pendingContent);
        int flushLength = pendingContent.length() - keep;
        if (flushLength <= 0) {
            return "";
        }
        String flush = pendingContent.substring(0, flushLength);
        pendingContent.delete(0, flushLength);
        return flush;
    }

    private static int firstToolTagIndex(CharSequence text) {
        int tools = indexOf(text, "<tools>");
        int toolCall = indexOf(text, "<tool_call>");
        if (tools == -1) {
            return toolCall;
        }
        if (toolCall == -1) {
            return tools;
        }
        return Math.min(tools, toolCall);
    }

    private static int longestToolTagPrefixSuffix(CharSequence text) {
        return Math.max(longestPrefixSuffix(text, "<tools>"), longestPrefixSuffix(text, "<tool_call>"));
    }

    private static int longestPrefixSuffix(CharSequence text, String prefix) {
        int max = Math.min(text.length(), prefix.length() - 1);
        for (int length = max; length > 0; length--) {
            boolean matches = true;
            for (int i = 0; i < length; i++) {
                if (text.charAt(text.length() - length + i) != prefix.charAt(i)) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                return length;
            }
        }
        return 0;
    }

    private static int indexOf(CharSequence text, String needle) {
        for (int i = 0; i <= text.length() - needle.length(); i++) {
            boolean matches = true;
            for (int j = 0; j < needle.length(); j++) {
                if (text.charAt(i + j) != needle.charAt(j)) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                return i;
            }
        }
        return -1;
    }

    private void sendStreamToken(SseEmitter emitter, AtomicBoolean closed, UUID sessionId, String token) {
        if (closed.get() || token == null || token.isEmpty()) {
            return;
        }
        sendStreamEvent(emitter, closed, streamDelta(sessionId, token));
    }

    private void sendStreamEvent(SseEmitter emitter, AtomicBoolean closed, Object event) {
        if (closed.get()) {
            return;
        }
        try {
            if (event instanceof SseEmitter.SseEventBuilder builder) {
                emitter.send(builder);
            } else {
                emitter.send(event);
            }
        } catch (IOException | RuntimeException e) {
            closed.set(true);
            LOGGER.debug("stream emitter closed before send completed", e);
        }
    }

    /**
     * Maps internal sampler-return data into the OpenAI-compatible logprobs response shape when the
     * generator captured top-logprob candidates. Returns {@code null} when logprobs were not requested.
     */
    static CreateChatCompletionResponseChoicesInnerLogprobs toLogProbs(Response response) {
        if (response.samplerReturns == null || response.samplerReturns.isEmpty()) {
            return null;
        }
        boolean hasAny = response.samplerReturns.stream().anyMatch(s -> s.getTopNLogProbs().isPresent());
        if (!hasAny) {
            return null;
        }
        CreateChatCompletionResponseChoicesInnerLogprobs out = new CreateChatCompletionResponseChoicesInnerLogprobs();
        response.samplerReturns.forEach(s -> out.addContentItem(toTokenLogprob(s)));
        return out;
    }

    static ChatCompletionTokenLogprob toTokenLogprob(SamplerReturn samplerReturn) {
        ChatCompletionTokenLogprob token = new ChatCompletionTokenLogprob()
                .token(firstTokenString(samplerReturn))
                .logprob(BigDecimal.valueOf(chosenLogProb(samplerReturn)))
                .topLogprobs(toTopLogprobs(samplerReturn));
        return token;
    }

    static List<ChatCompletionTokenLogprobTopLogprobsInner> toTopLogprobs(SamplerReturn samplerReturn) {
        if (samplerReturn.getTopNLogProbs().isEmpty()) {
            return List.of();
        }
        ArrayList<IndexValueToken> ranked = new ArrayList<>(samplerReturn.getTopNLogProbs().get());
        ranked.sort((a, b) -> Float.compare(b.value, a.value));
        ArrayList<ChatCompletionTokenLogprobTopLogprobsInner> mapped = new ArrayList<>(ranked.size());
        for (IndexValueToken candidate : ranked) {
            mapped.add(new ChatCompletionTokenLogprobTopLogprobsInner()
                    .token(candidate.token)
                    .logprob(BigDecimal.valueOf(candidate.logProb)));
        }
        return mapped;
    }

    private static String firstTokenString(SamplerReturn samplerReturn) {
        return samplerReturn.getTopNLogProbs()
                .flatMap(queue -> queue.stream().filter(t -> t.index == samplerReturn.getToken()).findFirst())
                .map(t -> t.token)
                .orElse(String.valueOf(samplerReturn.getToken()));
    }

    private static double chosenLogProb(SamplerReturn samplerReturn) {
        return samplerReturn.getTopNLogProbs()
                .flatMap(queue -> queue.stream().filter(t -> t.index == samplerReturn.getToken()).findFirst())
                .map(t -> (double) t.logProb)
                .orElse(-9999.0d);
    }

    private ObjectNode streamComplete(UUID sessionId, io.teknek.deliverance.generator.FinishReason finishReason){
        ObjectNode response = JsonUtils.om.createObjectNode();
        response.put("id", sessionId.toString());
        ArrayNode choices = response.putArray("choices");
        ObjectNode choice = choices.addObject();
        choice.put("index", 0);
        choice.set("delta", JsonUtils.om.createObjectNode());
        choice.put("finish_reason", streamFinishReason(finishReason));
        return response;
    }

    private ObjectNode streamDelta(UUID sessionId, String t){
        ObjectNode response = JsonUtils.om.createObjectNode();
        response.put("id", sessionId.toString());
        ArrayNode choices = response.putArray("choices");
        ObjectNode choice = choices.addObject();
        choice.put("index", 0);
        ObjectNode delta = choice.putObject("delta");
        delta.put("content", t);
        return response;
    }

    private ObjectNode streamToolCallDelta(UUID sessionId, int index, ToolCall toolCall) throws JsonProcessingException {
        ObjectNode response = JsonUtils.om.createObjectNode();
        response.put("id", sessionId.toString());
        ArrayNode choices = response.putArray("choices");
        ObjectNode choice = choices.addObject();
        choice.put("index", 0);
        ObjectNode delta = choice.putObject("delta");
        ArrayNode toolCalls = delta.putArray("tool_calls");
        ObjectNode call = toolCalls.addObject();
        call.put("index", index);
        call.put("id", toolCall.getId());
        call.put("type", "function");
        ObjectNode function = call.putObject("function");
        function.put("name", toolCall.getName());
        function.put("arguments", JsonUtils.om.writeValueAsString(toolCall.getParameters()));
        return response;
    }

    private String streamFinishReason(
            io.teknek.deliverance.generator.FinishReason finishReason) {
        return switch (finishReason) {
            case MAX_TOKENS -> "length";
            case TOOL_CALLS -> "tool_calls";
            case STOP_TOKEN -> "stop";
            case CONTENT_FILTER, FUNCTION_CALL -> throw new UnreachableException("not implemented");
            case null -> "stop";
        };
    }

    private ResponseEntity<ErrorResponse> badRequest(String message) {
        return new ResponseEntity<>(new ErrorResponse().error(new Error()
                .code(Integer.toString(HttpStatus.BAD_REQUEST.value()))
                .message(message)
                .type("bad_request")), HttpStatus.BAD_REQUEST);
    }

    private void debugRequest(CreateChatCompletionRequest request) {
        if (!debugChatRequest) {
            return;
        }
        LOGGER.info("chat.request model={} messages={} tools={} ntokens={} max_tokens={} temperature={}",
                request.getModel(),
                request.getMessages() == null ? 0 : request.getMessages().size(),
                request.getTools() == null ? 0 : request.getTools().size(),
                request.getNtokens(),
                request.getMaxTokens(),
                request.getTemperature());
        if (request.getMessages() != null) {
            for (int i = 0; i < request.getMessages().size(); i++) {
                LOGGER.info("chat.message[{}] {}", i, preview(String.valueOf(request.getMessages().get(i)), 512));
            }
        }
    }

    private void debugPrompt(String prompt) {
        if (!debugChatRequest) {
            return;
        }
        LOGGER.info("chat.prompt chars={} preview={}", prompt.length(), preview(prompt, 1200));
    }

    private void debugElapsed(String label, long startNanos) {
        if (!debugChatRequest) {
            return;
        }
        LOGGER.info("{} elapsed_ms={}", label, (System.nanoTime() - startNanos) / 1_000_000.0);
    }

    private static String preview(String value, int limit) {
        String sanitized = value.replace("\n", "\\n").replace("\r", "\\r");
        return sanitized.length() <= limit ? sanitized : sanitized.substring(0, limit) + "...";
    }

    /**
     *
     * @param builder
     * @param messages
     * @return entity only IF the message is invalid null = good
     */
    private ResponseEntity<Object> messagesToBuilder(PromptSupport.Builder builder, List<ChatCompletionRequestMessage> messages){
        for (ChatCompletionRequestMessage m : messages) {
            if (m.getActualInstance() instanceof ChatCompletionRequestUserMessage) {
                ChatCompletionRequestUserMessageContent content = m.getChatCompletionRequestUserMessage().getContent();
                if (content.getActualInstance() instanceof String) {
                    builder.addUserMessage(content.getString());
                } else {
                    for (ChatCompletionRequestMessageContentPart p : content.getListChatCompletionRequestMessageContentPart()) {
                        if (p.getActualInstance() instanceof ChatCompletionRequestMessageContentPartText) {
                            builder.addUserMessage(p.getChatCompletionRequestMessageContentPartText().getText());
                        } else {
                            // We don't su  pport other types of content... yet...
                            return new ResponseEntity<>(HttpStatus.NOT_IMPLEMENTED);
                        }
                    }
                }
            } else if (m.getActualInstance() instanceof ChatCompletionRequestSystemMessage) {
                builder.addSystemMessage(m.getChatCompletionRequestSystemMessage().getContent());
            } else if (m.getActualInstance() instanceof ChatCompletionRequestAssistantMessage) {
                builder.addAssistantMessage(m.getChatCompletionRequestAssistantMessage().getContent());
            } else if (m.getActualInstance() instanceof ChatCompletionRequestToolMessage) {
                ChatCompletionRequestToolMessage toolMessage = m.getChatCompletionRequestToolMessage();
                builder.addToolResult(io.teknek.deliverance.safetensors.prompt.ToolResult.from("tool",
                        toolMessage.getToolCallId(), toolMessage.getContent()));
            } else {
                return new ResponseEntity<>(HttpStatus.NOT_IMPLEMENTED);
            }
        }
        return null;
    }

}
