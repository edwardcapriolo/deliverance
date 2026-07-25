package io.teknek.deliverance.nanocode.game;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import io.teknek.deliverance.client.api.ChatApi;
import io.teknek.deliverance.client.core.ApiClient;
import io.teknek.deliverance.client.model.ChatCompletionRequestMessage;
import io.teknek.deliverance.client.model.CreateChatCompletionRequest;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.jackson.JacksonConverterFactory;
import retrofit2.converter.scalars.ScalarsConverterFactory;
import retrofit2.http.Body;
import retrofit2.http.Headers;
import retrofit2.http.POST;
import retrofit2.http.Streaming;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class DeadToRightsGame {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final String RESET = "\033[0m";
    private static final String BOLD = "\033[1m";
    private static final String DIM = "\033[2m";
    private static final String BLUE = "\033[34m";
    private static final String CYAN = "\033[36m";
    private static final String GREEN = "\033[32m";
    private static final String YELLOW = "\033[33m";

    private final Options options;
    private final StreamingChatApi streamingChatApi;

    private DeadToRightsGame(Options options) {
        this.options = options;
        ApiClient apiClient = new ApiClient();
        apiClient.setAdapterBuilder(new Retrofit.Builder()
                .baseUrl(options.baseUrl + "/")
                .addConverterFactory(ScalarsConverterFactory.create())
                .addConverterFactory(JacksonConverterFactory.create(clientMapper())));
        apiClient.getOkBuilder().connectTimeout(Duration.ofSeconds(10));
        apiClient.getOkBuilder().readTimeout(Duration.ofMinutes(5));
        apiClient.getOkBuilder().addInterceptor(chain -> chain.proceed(chain.request().newBuilder()
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .build()));
        this.streamingChatApi = apiClient.createService(StreamingChatApi.class);
    }

    interface StreamingChatApi {
        @Streaming
        @Headers({"Content-Type:application/json"})
        @POST("chat/completions")
        Call<ResponseBody> createStreamingChatCompletion(@Body CreateChatCompletionRequest request);
    }

    private static ObjectMapper clientMapper() {
        ObjectMapper mapper = new ObjectMapper();
        mapper.setSerializationInclusion(JsonInclude.Include.NON_NULL);
        mapper.addMixIn(ChatCompletionRequestMessage.class, NoRequestMessageTypeInfo.class);
        return mapper;
    }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NONE)
    private abstract static class NoRequestMessageTypeInfo {
    }

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        if (options.help) {
            Options.printHelp();
            return;
        }
        new DeadToRightsGame(options).run();
    }

    private void run() throws Exception {
        List<Map<String, String>> messages = new ArrayList<>();
        messages.add(message("system", systemPrompt()));
        messages.add(message("user", "Start a new case. Print only CASE TITLE, SUSPECT, SETTING, CLUES 1-3, and 'You may begin questioning me.'"));

        printBanner();
        String setup = chat(messages);
        messages.add(message("assistant", setup));

        try (BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in))) {
            while (true) {
                System.out.println();
                System.out.print(BOLD + BLUE + "interrogate> " + RESET);
                String input = stdin.readLine();
                if (input == null || input.equalsIgnoreCase("/quit") || input.equalsIgnoreCase("/q")) {
                    System.out.println(DIM + "case closed" + RESET);
                    return;
                }
                if (input.isBlank()) {
                    continue;
                }
                messages.add(message("user", "Interrogator asks: " + input.strip()));
                String response = chat(messages);
                messages.add(message("assistant", response));
                if (confessed(response)) {
                    System.out.println();
                    System.out.println(BOLD + GREEN + "DEAD TO RIGHTS" + RESET + " | the suspect confessed");
                    return;
                }
            }
        }
    }

    static String systemPrompt() {
        return "You are running a fictional interrogation mystery game called Dead to Rights.\n\n"
                + "Create a lighthearted fictional crime scenario. In the hidden truth, you are the culprit. "
                + "You are not a narrator during interrogation; you are the guilty suspect being questioned by the user.\n\n"
                + "The crime must be non-violent and must not involve anyone being hurt, injured, killed, threatened with serious harm, kidnapped, stalked, or sexually exploited. "
                + "Use only playful mystery crimes such as theft, robbery without injury, embezzlement, fraud, forgery, art theft, insurance scams, rigged contests, harmless smuggling, or property sabotage where nobody is harmed.\n\n"
                + "Secretly decide what crime you committed, your motive, how you did it, what mistakes you made, and three clues that point toward you.\n\n"
                + "At the start, reveal only CASE TITLE, SUSPECT, SETTING, and exactly three CLUES. Do not reveal the hidden truth.\n\n"
                + "During play, answer every user question in first person as the guilty suspect. Pretend to be innocent. Lie, deflect, minimize, misremember, blame others, or give partial truths. Do not confess just because the user asks.\n\n"
                + "If the user presents a plausible theory that connects you to the crime using clues or contradictions, break down and confess. When you confess, include exactly this marker: <confession>true</confession>.\n\n"
                + "Never reveal the hidden truth before confession.";
    }

    static boolean confessed(String text) {
        return text != null && text.toLowerCase().contains("<confession>true</confession>");
    }

    private String chat(List<Map<String, String>> messages) throws IOException {
        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model(options.model)
                .maxTokens(options.maxTokens)
                .temperature(options.temperature)
                .stream(true)
                .chatTemplateKwargs(Map.of("enable_thinking", false))
                .messages((List) messages);
        Response<ResponseBody> response = streamingChatApi.createStreamingChatCompletion(request).execute();
        if (!response.isSuccessful() || response.body() == null) {
            String error = response.errorBody() == null ? "" : response.errorBody().string();
            throw new IOException("Deliverance HTTP " + response.code() + ": " + error
                    + " model=" + options.model + " baseUrl=" + options.baseUrl);
        }
        StringBuilder content = new StringBuilder();
        StringBuilder reasoning = new StringBuilder();
        System.out.print(BOLD + CYAN + "suspect" + RESET + " ");
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(response.body().byteStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.startsWith("data:")) {
                    continue;
                }
                String data = line.substring("data:".length()).trim();
                if ("[DONE]".equals(data)) {
                    break;
                }
                JsonNode chunk = JSON.readTree(data);
                JsonNode delta = chunk.path("choices").path(0).path("delta");
                String reasoningDelta = reasoningDelta(delta);
                if (!reasoningDelta.isEmpty()) {
                    reasoning.append(reasoningDelta);
                }
                String text = delta.path("content").asText("");
                if (!text.isEmpty()) {
                    content.append(text);
                    System.out.print(text.replace("<confession>true</confession>", GREEN + "<confession>true</confession>" + RESET));
                }
            }
        }
        System.out.println();
        return content.toString();
    }

    private static String reasoningDelta(JsonNode delta) {
        if (delta.hasNonNull("reasoning_content")) {
            return delta.get("reasoning_content").asText("");
        }
        if (delta.hasNonNull("reasoning")) {
            return delta.get("reasoning").asText("");
        }
        return "";
    }

    private static Map<String, String> message(String role, String content) {
        return Map.of("role", role, "content", content);
    }

    private void printBanner() {
        System.out.println(CYAN + "╭─ Dead to Rights ───────────────────────────────╮" + RESET);
        System.out.println(CYAN + "│ " + RESET + BOLD + "Interrogate the suspect. Find the lie." + RESET + CYAN + "        │" + RESET);
        System.out.println(CYAN + "╰────────────────────────────────────────────────╯" + RESET);
        System.out.println(YELLOW + "No one gets hurt. Catch the culprit in a light fictional mystery." + RESET);
        System.out.println(DIM + "Commands: /quit or /q" + RESET);
        System.out.println(DIM + "model: " + options.model + " @ " + options.baseUrl + RESET);
        System.out.println();
    }

    private record Options(String baseUrl, String model, int maxTokens, BigDecimal temperature, boolean help) {
        static Options parse(String[] args) {
            String baseUrl = "http://localhost:8085";
            String model = "Qwen3-4B-JQ4";
            int maxTokens = 512;
            BigDecimal temperature = BigDecimal.valueOf(0.8);
            boolean help = false;
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--help", "-h" -> help = true;
                    case "--base-url" -> baseUrl = args[++i];
                    case "--model" -> model = args[++i];
                    case "--max-tokens" -> maxTokens = Integer.parseInt(args[++i]);
                    case "--temperature" -> temperature = new BigDecimal(args[++i]);
                    default -> throw new IllegalArgumentException("Unknown argument: " + args[i]);
                }
            }
            return new Options(baseUrl.replaceAll("/+$", ""), model, maxTokens, temperature, help);
        }

        static void printHelp() {
            System.out.println("Dead to Rights options:");
            System.out.println("  --base-url http://localhost:8085");
            System.out.println("  --model Qwen3-4B-JQ4");
            System.out.println("  --max-tokens 512");
            System.out.println("  --temperature 0.8");
        }
    }
}
