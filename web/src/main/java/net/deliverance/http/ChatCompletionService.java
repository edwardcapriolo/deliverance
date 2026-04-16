package net.deliverance.http;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.model.*;
import io.teknek.deliverance.model.Error;
import io.teknek.deliverance.safetensors.prompt.Function;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.safetensors.prompt.Tool;
import io.teknek.dysfx.Either;
import org.springframework.http.HttpStatus;

import java.util.*;

public class ChatCompletionService {
    public static Either<Error, PreparedRequest> mapRequest(Map<String, String> headers, AbstractModel model,
                                                            CreateChatCompletionRequest request) {
        Optional<PromptSupport> ps = model.promptSupport();
        if (ps.isEmpty()) {
            return Either.Left(new Error().code(HttpStatus.BAD_REQUEST.value() + "")
                    .message("This model does mot have prompt support"));
        }
        PromptSupport.Builder builder = ps.get().builder();
        GeneratorParameters params = new GeneratorParameters();
        if (request.getTemperature() != null) {
            params.withTemperature(request.getTemperature().floatValue());
        }
        try {
            //There are many possible data errors in here
            if (request.getTools() != null) {
                for (ChatCompletionTool chatCompletionTool : request.getTools()) {
                    Tool t = convert(chatCompletionTool);
                    builder.addToolItem(t);
                }
            }
        } catch (RuntimeException e) {
            return Either.Left(new Error().code(HttpStatus.BAD_REQUEST.value() + "")
                    .message("An error happened mapping tools" + e.getMessage()));
        }
        if (request.getTopP()!= null){
            params.withTopP(request.getTopP().floatValue());
        }
        if (request.getTopK()!=null){
            params.withTopK(request.getTopK().floatValue());
        }
        if (request.getXtcThreshold()!=null){
            params.withXtcThreshold(request.getXtcThreshold().floatValue());
        }
        if (request.getXtcProbability()!= null){
            params.withXtcProbability(request.getXtcProbability().floatValue());
        }
        if (request.getNtokens() != null) {
            params.withNtokens(request.getNtokens());
        }
        if (request.getSeed() != null) {
            params.withSeed(request.getSeed());
        }
        if (request.getStop() != null) {
            CreateChatCompletionRequestStop stop = request.getStop();
            if (stop.getActualInstance() instanceof String) {
                params.withStopWords(Collections.singletonList(stop.getString()));
            } else {
                params.withStopWords(stop.getListString());
            }
        }
        if (request.getChatTemplate() != null) {
            builder.useChatTemplate(request.getChatTemplate());
        }
        if (request.getMaxTokens() != null) {
            params.withMaxTokens(request.getMaxTokens());
        }
        for (ChatCompletionRequestMessage m : request.getMessages()) {
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
                                    .message("User messages must be type text " + m.getActualInstance()));
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

    /**
     * This is a frustrating method as we are really mapping two identical types through a 1-1 code transforrmation.
     *
     * @param tool the input tool in the api format
     * @return the same inforamtion in the internal (deliverance prompt) format
     */
    public static Tool convert(ChatCompletionTool tool) {
        FunctionObject f = tool.getFunction();
        Function.Builder builder = Function.builder();
        builder.name(f.getName());
        builder.description(f.getDescription());

        List<String> required = (List<String>) f.getParameters().get("required");
        if (required == null) {
            required = new ArrayList<>();
        }
        Map<String, Object> properties = (Map<String, Object>) f.getParameters().get("properties");
        if (properties == null) {
            return Tool.from(builder.build());
        }
        for (Map.Entry<String, Object> entry : properties.entrySet()) {
            String name = entry.getKey();
            Map<String, Object> value = (Map<String, Object>) entry.getValue(); //{ "type": "string", "description": "State abbreviation" },
            String type = (String) value.get("type");
            String description = (String) value.get("description");
            builder.addParameter(name, type, description, required.contains(name));

        }
        return Tool.from(builder.build());
    }

}
