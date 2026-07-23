package io.teknek.deliverance.antares;

import java.io.IOException;
import java.util.List;
import java.util.function.Consumer;

interface CompletionClient {
    String complete(List<Message> messages, Consumer<String> onChunk) throws IOException;
}
