package net.deliverance.http;

import io.teknek.deliverance.model.CreateChatCompletionRequest;
import io.teknek.deliverance.model.Error;
import io.teknek.dysfx.Either;

public interface PreGenerateSlot {
    /**
     *
     * @param originalRequest The payload sent to the endpoint
     * @param preparedRequest The request prepared for the llm
     * @return Error (Left) when you wish to block the request (Right) PreparedRequest to continue
     */
    Either<Error, PreparedRequest> handle(CreateChatCompletionRequest originalRequest, PreparedRequest preparedRequest);
}
