package net.deliverance.http;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.*;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
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
        GeneratorParameters params = new GeneratorParameters().withTemperature(0.0f);

        AtomicInteger index = new AtomicInteger(0);
        if (request.getStream() != null && request.getStream()) {

            SseEmitter emitter = new SseEmitter(-1L);
            CompletableFuture.supplyAsync(
                    () -> model.generate(sessionId, builder.build(), params, (t, f) -> CompletableFuture.supplyAsync(() -> {
                        try {
                            emitter.send(
                                    new CreateChatCompletionStreamResponse().id(sessionId.toString())
                                            .choices(
                                                    List.of(
                                                            new CreateChatCompletionStreamResponseChoicesInner().index(index.getAndIncrement())
                                                                    .delta(new ChatCompletionStreamResponseDelta().content(t))
                                                    )
                                            )
                            );
                        } catch (IOException e) {
                            emitter.completeWithError(e);
                        }
                        return null;
                    }))
            ).handle((r, ex) -> {
                try {
                    emitter.send(
                            new CreateChatCompletionStreamResponse().id(sessionId.toString())
                                    .choices(
                                            List.of(
                                                    new CreateChatCompletionStreamResponseChoicesInner().finishReason(
                                                            CreateChatCompletionStreamResponseChoicesInner.FinishReasonEnum.STOP
                                                    ).delta(new ChatCompletionStreamResponseDelta().content(""))
                                            )
                                    )
                    );

                    emitter.complete();



                } catch (IOException e) {
                    emitter.completeWithError(e);
                }

                return null;
            });

            return emitter;
        } else {
            Response resp = model.generate(UUID.randomUUID(), builder.build(), params, (s, aFloat) -> {});
            CreateChatCompletionResponse out = new CreateChatCompletionResponse().id("ok")
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
}
