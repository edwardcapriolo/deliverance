package net.deliverance.http;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.model.*;
import io.teknek.deliverance.model.Error;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.dysfx.Either;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import java.util.Map;
import java.util.Optional;

public class ChatCompletionService {
    public static Either<Error,PreparedRequest> mapRequest(Map<String, String> headers, AbstractModel model,
                                      CreateChatCompletionRequest request){
        System.err.println(request);
        Optional<PromptSupport> ps = model.promptSupport();
        if (ps.isEmpty()){
            return Either.Left(new Error().code(HttpStatus.BAD_REQUEST.value() + "")
                    .message("This model does mot have prompt support"));
        }
        PromptSupport.Builder builder = ps.get().builder();
        GeneratorParameters params = new GeneratorParameters();
        if (request.getTemperature() != null){
            params.withTemperature(request.getTemperature().floatValue());
        }

        if (request.getMaxTokens() != null){
            params.withNtokens(request.getMaxTokens());
        }
        if (request.getSeed() != null){
            params.withSeed(request.getSeed());
        }
        if(request.getStop() != null){
            CreateChatCompletionRequestStop stop = request.getStop();
            //deal with polymorphic crap
        }
        if(request.getChatTemplate() != null){
            builder.useChatTemplate(request.getChatTemplate());
        }
        for (ChatCompletionRequestMessage m : request.getMessages()){
            if (m.getActualInstance() instanceof ChatCompletionRequestUserMessage) {
                ChatCompletionRequestUserMessageContent content = m.getChatCompletionRequestUserMessage().getContent();
                if (content.getActualInstance() instanceof String) {
                    builder.addUserMessage(content.getString());
                } else {
                    for (ChatCompletionRequestMessageContentPart p : content.getListChatCompletionRequestMessageContentPart()) {
                        if (p.getActualInstance() instanceof ChatCompletionRequestMessageContentPartText) {
                            builder.addUserMessage(p.getChatCompletionRequestMessageContentPartText().getText());
                        } else {
                            return Either.Left(new Error().code(HttpStatus.BAD_REQUEST.value() + "")
                                    .message("User messages must be type text"+m.getActualInstance()));
                        }
                    }
                }
            } else if (m.getActualInstance() instanceof ChatCompletionRequestSystemMessage) {
                builder.addSystemMessage(m.getChatCompletionRequestSystemMessage().getContent());
            } else if (m.getActualInstance() instanceof ChatCompletionRequestAssistantMessage) {
                builder.addAssistantMessage(m.getChatCompletionRequestAssistantMessage().getContent());
            } else {
                return Either.Left(new Error().code(HttpStatus.BAD_REQUEST.value() + "")
                        .message("Could not handle " + m.getActualInstance()));
            }
        }


        return Either.Right(new PreparedRequest(builder, params));
    }
}
