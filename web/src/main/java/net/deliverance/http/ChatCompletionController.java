package net.deliverance.http;

import com.fasterxml.jackson.core.JsonProcessingException;
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
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;


@RestController
public class ChatCompletionController {
    private static final Logger LOGGER = LoggerFactory.getLogger(ChatCompletionController.class);

    private static final String DELIVERANCE_SESSION_HEADER = "X-Deliverance-Session";

    @Autowired
    private Map<MultiModelConfig,AbstractModel> models;

    private Optional<Map.Entry<MultiModelConfig, AbstractModel>> findModel(String name){
        return models.entrySet().stream()
                .filter(x-> x.getKey().getModelName()
                        .equalsIgnoreCase(name)).findFirst();
    }

    private PreGenerateSlot slot;
    private final boolean debugChatRequest;

    public ChatCompletionController(Optional<PreGenerateSlot> slot,
            @Value("${deliverance.debug.chat-request:false}") boolean debugChatRequest){
        this.slot = slot.orElse((x,y) -> Either.Right(y));
        this.debugChatRequest = debugChatRequest;
    }

    @RequestMapping(method = RequestMethod.POST, value = "/chat/completions", produces = { "application/json",
            "text/event-stream" }, consumes = { "application/json" })
    Object createChatCompletion(@RequestHeader Map<String, String> headers,
                                @RequestBody CreateChatCompletionRequest request) {
        Optional<Map.Entry<MultiModelConfig, AbstractModel>> z = findModel(request.getModel());
        if (z.isEmpty()){
            return badRequest("model not found " + request.getModel());
        }
        AbstractModel model = z.get().getValue();
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

        //This is the older stuff lets clean it out
        //its here to support streaming which needs to be considered separately
        List<ChatCompletionRequestMessage> messages = request.getMessages();
        UUID id = UUID.randomUUID();
        if (headers.containsKey(DELIVERANCE_SESSION_HEADER)) {
            try {
                id = UUID.fromString(headers.get(DELIVERANCE_SESSION_HEADER));
            } catch (IllegalArgumentException e) {
                return new ResponseEntity<>(HttpStatus.BAD_REQUEST);
            }
        }
        UUID sessionId = id;
        PromptSupport.Builder builder = model.promptSupport().get().builder();
        ResponseEntity<Object> result = messagesToBuilder(builder, messages);
        if (result != null){
            return result;
        }
        GeneratorParameters params = new GeneratorParameters().withTemperature(0.1f);
        AtomicInteger index = new AtomicInteger(0);
        LOGGER.info("submitted prompt {}", builder.build());
        if (request.getStream() != null && request.getStream()) {
            SseEmitter emitter = new SseEmitter(-1L);
            CompletableFuture<Response> generate = CompletableFuture.supplyAsync( () -> {
                return model.generate(sessionId, builder.build(), params, (int next, String tok, String token, float f) -> {
                            try {
                                emitter.send( messageDelta(sessionId, token, index));
                            } catch (IOException  | RuntimeException e) {
                                LOGGER.error("emitter issue", e);
                                emitter.completeWithError(e);
                            }
                        }
                );
            }).handle((result2, throwable) -> {
                if (throwable == null){
                    try {
                        emitter.send(sendComplete(sessionId, index));
                    } catch (IOException | RuntimeException e) {
                        LOGGER.error("emitter issue", e);
                        throw new RuntimeException(e);
                    }
                    emitter.complete();
                } else {
                    emitter.completeWithError(throwable);
                }
                return result2;
            });
            return emitter;
        }
        /*
        else {
            Response resp = model.generate(UUID.randomUUID(), builder.build(), params, (int next, String rok, String s, float aFloat) -> {});
            CreateChatCompletionResponse out = new CreateChatCompletionResponse().id(sessionId.toString())
                    .choices(
                            List.of(
                                    new CreateChatCompletionResponseChoicesInner().finishReason(
                                            CreateChatCompletionResponseChoicesInner.FinishReasonEnum.STOP
                                    ).message(new ChatCompletionResponseMessage().content(resp.responseText))
                            )
                    );
            return new ResponseEntity<>(out, HttpStatus.OK);
        }*/
        throw new UnsupportedOperationException("Hit bottom");
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

    private CreateChatCompletionStreamResponse sendComplete(UUID sessionId, AtomicInteger index){
        return new CreateChatCompletionStreamResponse().id(sessionId.toString())
                .choices(
                        List.of(
                                new CreateChatCompletionStreamResponseChoicesInner().finishReason(
                                        CreateChatCompletionStreamResponseChoicesInner.FinishReasonEnum.STOP
                                ).delta(new ChatCompletionStreamResponseDelta().content(""))
                        )
                );

    }
    private CreateChatCompletionStreamResponse messageDelta(UUID sessionId, String t, AtomicInteger index){
        return new CreateChatCompletionStreamResponse().id(sessionId.toString())
                .choices(
                        List.of(
                                new CreateChatCompletionStreamResponseChoicesInner().index(index.getAndIncrement())
                                        .delta(new ChatCompletionStreamResponseDelta().content(t))));
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
