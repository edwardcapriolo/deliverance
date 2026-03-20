package io.teknek.deliverance.toolcallparser;

import io.teknek.deliverance.generator.Response;

/**
 * This is a messy little interface. We have a text stream of token from a model,
 * that text stream has turns. The turns are signified differently in different models.
 * Also the end user with ChatCompletion is looking for a certain shape.
 * We will punt on having a model here, that understands the model returned via http
 * Instead just do a simple parsing and move on.
 *
 */
public interface ToolCallParser {
    /**
     *
     * @param response the response directly from the AbstractModel
     * @return at least one message
     */
    MessageAndToolCall extract(Response response);
}
