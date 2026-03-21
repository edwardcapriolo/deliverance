package io.teknek.deliverance.toolcallparser;

import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.safetensors.prompt.ToolCall;

import java.util.Collections;
import java.util.List;

public class DefaultToolCallParser implements ToolCallParser {

    @Override
    public List<ToolCall> extract(Response response) {
        return Collections.emptyList();
    }
}
