package net.deliverance.http;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.*;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;


@RestController
public class DeliveranceController {
    private static final Logger LOGGER = LoggerFactory.getLogger(DeliveranceController.class);

    private static final String DELIVERANCE_SESSION_HEADER = "X-Deliverance-Session";
    @Autowired
    private AbstractModel model;


    @RequestMapping(method = RequestMethod.POST, value = "/chat/completions", produces = { "application/json",
            "text/event-stream" }, consumes = { "application/json" })
    Object createChatCompletion(@RequestHeader Map<String, String> headers,
                                @RequestBody CreateChatCompletionRequest request) {
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
        GeneratorParameters params = new GeneratorParameters().withTemperature(0.1f).withSalt(sessionId.toString());
        AtomicInteger index = new AtomicInteger(0);
        builder.addSystemMessage("generate correct answers");
        LOGGER.info("submitted prompt {}", builder.build());
        if (request.getStream() != null && request.getStream()) {
            SseEmitter emitter = new SseEmitter(-1L);
            CompletableFuture<Response> generate = CompletableFuture.supplyAsync( () -> {
                return model.generate(sessionId, builder.build(), params, (String token, Float f) -> {
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
        } else {
            Response resp = model.generate(UUID.randomUUID(), builder.build(), params, (s, aFloat) -> {});
            CreateChatCompletionResponse out = new CreateChatCompletionResponse().id(sessionId.toString())
                    .choices(
                            List.of(
                                    new CreateChatCompletionResponseChoicesInner().finishReason(
                                            CreateChatCompletionResponseChoicesInner.FinishReasonEnum.STOP
                                    ).message(new ChatCompletionResponseMessage().content(resp.responseText))
                            )
                    );
            return new ResponseEntity<>(out, HttpStatus.OK);
        }
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
                            // We don't support other types of content... yet...
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
