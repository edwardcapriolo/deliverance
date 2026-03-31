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
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
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

    public ChatCompletionController(Optional<PreGenerateSlot> slot){
        this.slot = slot.orElse((x,y) -> Either.Right(y));
    }

    @RequestMapping(method = RequestMethod.POST, value = "/chat/completions", produces = { "application/json",
            "text/event-stream" }, consumes = { "application/json" })
    Object createChatCompletion(@RequestHeader Map<String, String> headers,
                                @RequestBody CreateChatCompletionRequest request) {
        Optional<Map.Entry<MultiModelConfig, AbstractModel>> z = findModel(request.getModel());
        if (z.isEmpty()){
            throw new RuntimeException("model not found " + request.getModel());
        }
        AbstractModel model = z.get().getValue();

        if (request.getStream() == null || (request.getStream() != null && request.getStream() == false)) {
            Either<Error, PreparedRequest> bla = ChatCompletionService.mapRequest(headers, model, request);
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
                Response resp = model.generate(UUID.randomUUID(), ready.promptSupportBuilder().build(),
                        ready.generatorParameters(), new DoNothingGenerateEvent());
                CreateChatCompletionResponse response = new CreateChatCompletionResponse();
                List<ToolCall> tcs = model.getToolCallParser().extract(resp);
                CreateChatCompletionResponseChoicesInner choice = new CreateChatCompletionResponseChoicesInner()
                        .message(new ChatCompletionResponseMessage().content(resp.responseTextWithSpecialTokens))
                        .index(0);
                if (!tcs.isEmpty()) {
                    //We shouldn't need this but it seems like the behaivor is hadto model
                    choice.setFinishReason(CreateChatCompletionResponseChoicesInner.FinishReasonEnum.TOOL_CALLS);
                } else {
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
            } else {
                return new ResponseEntity<>(HttpStatus.NOT_IMPLEMENTED);
            }
        }
        return null;
    }

}
