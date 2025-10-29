package net.deliverance.http;

import io.teknek.deliverance.generator.Generator;
import io.teknek.deliverance.model.ChatCompletionRequestMessage;
import io.teknek.deliverance.model.CreateChatCompletionRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
public class DeliveranceController {

    @Autowired
    private Generator model;

    @RequestMapping(method = RequestMethod.POST, value = "/chat/completions", produces = { "application/json",
            "text/event-stream" }, consumes = { "application/json" })
    Object createChatCompletion(@RequestHeader Map<String, String> headers,
                                @RequestBody CreateChatCompletionRequest request) {

        List<ChatCompletionRequestMessage> messages = request.getMessages();

        return "yo";
    }
}
