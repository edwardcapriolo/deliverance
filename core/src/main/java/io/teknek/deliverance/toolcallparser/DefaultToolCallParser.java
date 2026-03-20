package io.teknek.deliverance.toolcallparser;

import io.teknek.deliverance.generator.Response;

public class DefaultToolCallParser implements ToolCallParser {

    @Override
    public MessageAndToolCall extract(Response response) {
        MessageAndToolCall toolCall = new MessageAndToolCall();
        toolCall.messages.add(new MessageDatum("assistant",
                response.responseTextWithSpecialTokens, null));
        return toolCall;
    }
}
