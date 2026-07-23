package io.teknek.deliverance.antares;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

final class OpenAiCompletionsClient implements CompletionClient {
    private static final MediaType JSON_MEDIA_TYPE = MediaType.get("application/json");
    private static final ObjectMapper JSON = new ObjectMapper();

    private final OkHttpClient client;
    private final String endpoint;
    private final String model;
    private final int maxTokens;
    private final float temperature;
    private final float topP;

    OpenAiCompletionsClient(String endpoint, String model, int maxTokens, float temperature, float topP) {
        this(endpoint, model, maxTokens, temperature, topP, new OkHttpClient.Builder()
                .readTimeout(Duration.ofMinutes(10))
                .callTimeout(Duration.ofMinutes(15))
                .build());
    }

    OpenAiCompletionsClient(String endpoint, String model, int maxTokens, float temperature, float topP,
            OkHttpClient client) {
        this.endpoint = endpoint.replaceAll("/+$", "");
        this.model = model;
        this.maxTokens = maxTokens;
        this.temperature = temperature;
        this.topP = topP;
        this.client = client;
    }

    String complete(List<Message> messages) throws IOException {
        return complete(messages, ignored -> {
        });
    }

    @Override
    public String complete(List<Message> messages, Consumer<String> onChunk) throws IOException {
        String prompt = AntaresPrompt.render(messages);
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("model", model);
        payload.put("prompt", prompt);
        payload.put("stream", true);
        payload.put("max_tokens", maxTokens);
        payload.put("temperature", temperature);
        payload.put("top_p", topP);
        payload.put("stop", List.of("<|end_of_text|>", "<|endoftext|>", "<|eot_id|>"));
        Request request = new Request.Builder()
                .url(resolveCompletionsUrl())
                .post(RequestBody.create(JSON.writeValueAsBytes(payload), JSON_MEDIA_TYPE))
                .build();
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String body = response.body() == null ? "" : response.body().string();
                throw new IOException("Completions request failed HTTP " + response.code() + ": " + body);
            }
            if (response.body() == null) {
                throw new IOException("Completions response had no body");
            }
            StringBuilder text = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(response.body().charStream())) {
                String line;
                while ((line = reader.readLine()) != null) {
                    String chunk = parseSseLine(line);
                    if (chunk == null) {
                        continue;
                    }
                    text.append(chunk);
                    onChunk.accept(chunk);
                }
            }
            return text.toString();
        }
    }

    private String resolveCompletionsUrl() {
        if (endpoint.endsWith("/completions")) {
            return endpoint;
        }
        if (endpoint.endsWith("/v1")) {
            return endpoint + "/completions";
        }
        return endpoint + "/v1/completions";
    }

    static String parseSse(String body) throws IOException {
        StringBuilder text = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new StringReader(body))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String chunk = parseSseLine(line);
                if (chunk != null) {
                    text.append(chunk);
                }
            }
        }
        return text.toString();
    }

    private static String parseSseLine(String line) throws IOException {
        if (!line.startsWith("data:")) {
            return null;
        }
        String data = line.substring("data:".length()).trim();
        if ("[DONE]".equals(data)) {
            return null;
        }
        JsonNode root = JSON.readTree(data);
        JsonNode choices = root.get("choices");
        if (choices == null || choices.isEmpty()) {
            return null;
        }
        JsonNode choice = choices.get(0);
        JsonNode chunk = choice.get("text");
        if (chunk != null && !chunk.isNull()) {
            return chunk.asText();
        }
        return null;
    }
}
