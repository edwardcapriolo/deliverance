package io.teknek.deliverance.toolcallparser;

import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.ResponseContext;
import io.teknek.deliverance.safetensors.prompt.ToolCall;

import java.util.Collections;
import java.util.List;
import java.util.Optional;

public class MixtralToolCallParser implements ToolCallParser {
    //This is the promptTemplate not the result
    public static final String HEADER = "[AVAILABLE_TOOLS]";
    public static final String TRAILER = "[/AVAILABLE_TOOLS]";
    @Override
    public List<ToolCall> extract(Response response) {
        int x = response.responseTextWithSpecialTokens.indexOf(HEADER);
        if (x == -1){
            return Collections.emptyList();
        }
        String inside = response.responseTextWithSpecialTokens.substring(x + HEADER.length(),
                response.responseTextWithSpecialTokens.indexOf(TRAILER));
        return null;
    }

    @Override
    public Optional<Response> shouldEndTurn(ResponseContext response, int length) {
        return Optional.empty();
    }
}
