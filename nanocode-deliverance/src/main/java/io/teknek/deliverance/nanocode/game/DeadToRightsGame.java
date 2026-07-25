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
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;

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
    private final int seed;
    private final CaseVariation caseVariation;
    private String publicCase = "";
    private String hiddenTruth = "";

    private DeadToRightsGame(Options options) {
        this.options = options;
        this.seed = (int) (System.currentTimeMillis() & 0x7fffffff);
        this.caseVariation = CaseVariation.random(seed);
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
        messages.add(message("user", caseVariation.setupPrompt()));

        printBanner();
        System.out.println(DIM + "A crime has been committed. The investigating officer has brought in the suspect "
                + "and is preparing the details for you. The suspect is waiting in the interrogation room. "
                + "You are our best interrogator; we need you to get in there and get a confession!" + RESET);
        String setup = chat(messages, false);
        if (setup.isBlank()) {
            throw new IOException("The model produced no visible case opening. Try a larger --max-tokens value or type /showthink on the next run to inspect hidden reasoning.");
        }
        publicCase = extractTag(setup, "public");
        hiddenTruth = extractTag(setup, "hidden_truth");
        if (publicCase.isBlank() || hiddenTruth.isBlank()) {
            throw new IOException("The model did not produce a valid tagged case file. Expected <public> and <hidden_truth> sections. Raw setup: " + setup);
        }
        System.out.println(BOLD + CYAN + "suspect" + RESET + " " + publicCase.strip());
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
                if (input.equalsIgnoreCase("/reveal")) {
                    System.out.println();
                    System.out.println(BOLD + YELLOW + "HIDDEN TRUTH" + RESET);
                    System.out.println(hiddenTruth.strip());
                    return;
                }
                if (input.equalsIgnoreCase("/case")) {
                    System.out.println(publicCase.strip());
                    continue;
                }
                if (input.isBlank()) {
                    continue;
                }
                messages.add(message("user", "Interrogator asks: " + input.strip()
                        + "\nAnswer visibly as the suspect. Do not leave the visible answer blank."));
                String response = chat(messages, true);
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
                + "The crime must be non-violent. Use only these crime types: theft, embezzlement, or forgery.\n\n"
                + "Secretly decide what crime you committed, your motive, how you did it, what mistakes you made, and three clues that point toward you.\n\n"
                + "At the start, reveal only CASE TITLE, SUSPECT, SETTING, and exactly three CLUES. Do not reveal the hidden truth.\n\n"
                + "During play, answer every user question in first person as the guilty suspect. Pretend to be innocent, but keep the mystery fun: give specific, useful details, partial truths, suspicious excuses, and small contradictions the user can follow up on. Do not repeatedly say only that you were not involved. If asked about an object, place, person, time, or motive, invent a concrete answer that fits the hidden truth while still trying to deflect blame. Do not confess just because the user asks.\n\n"
                + "If the user presents a plausible theory that connects you to the crime using clues or contradictions, break down and confess. "
                + "When you confess, explicitly admit guilt in first person before the marker, for example: 'I confess. I took it because...'. "
                + "Explain why you did it. Only then include exactly this marker: <confession>true</confession>.\n\n"
                + "Never reveal the hidden truth before confession.";
    }

    static boolean confessed(String text) {
        return text != null && text.toLowerCase().contains("<confession>true</confession>");
    }

    private String chat(List<Map<String, String>> messages) throws IOException {
        return chat(messages, true);
    }

    private String chat(List<Map<String, String>> messages, boolean printOutput) throws IOException {
        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model(options.model)
                .maxTokens(options.maxTokens)
                .seed(seed)
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
        if (printOutput) {
            System.out.print(BOLD + CYAN + "suspect" + RESET + " ");
        }
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
                    if (printOutput) {
                        System.out.print(text.replace("<confession>true</confession>", GREEN + "<confession>true</confession>" + RESET));
                    }
                }
            }
        }
        if (printOutput) {
            System.out.println();
        }
        return content.toString();
    }

    static String extractTag(String text, String tagName) {
        if (text == null) {
            return "";
        }
        String open = "<" + tagName + ">";
        String close = "</" + tagName + ">";
        String lower = text.toLowerCase();
        int start = lower.indexOf(open.toLowerCase());
        if (start < 0) {
            return "";
        }
        int contentStart = start + open.length();
        int end = lower.indexOf(close.toLowerCase(), contentStart);
        if (end < 0) {
            return "";
        }
        return text.substring(contentStart, end).strip();
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
        System.out.println(DIM + "Commands: /case, /reveal, /quit, /q" + RESET);
        System.out.println(DIM + "model: " + options.model + " @ " + options.baseUrl + " | seed=" + seed
                + " | " + caseVariation.shortLabel() + RESET);
        System.out.println();
    }

    private record CaseVariation(String id, String setting, String crime, String object, String suspectRole,
            String clueStyle) {
        private static CaseVariation random(int seed) {
            Random random = new Random((((long) seed) << 32) ^ System.nanoTime());
            return new CaseVariation(
                    UUID.randomUUID().toString(),
                    pick(random, "bookshop", "botanical conservatory", "museum fundraiser", "small-town bakery",
                            "community theater", "antique map fair", "hotel lost-and-found", "university archive",
                            "radio station", "local chess club"),
                    pick(random, "theft", "embezzlement", "forgery"),
                    pick(random, "rare stamp", "silver trophy", "donation ledger", "antique violin", "secret sauce recipe",
                            "signed first edition", "festival cash box", "prototype gadget", "theater prop crown", "archive key"),
                    pick(random, "bookkeeper", "assistant curator", "stage manager", "night clerk", "contest judge",
                            "catering manager", "archivist", "auction assistant", "maintenance lead", "club treasurer"),
                    pick(random, "physical clue", "paper trail", "timeline contradiction", "odd smell", "misplaced key",
                            "changed receipt", "muddy footprint", "rewritten note", "camera blind spot", "overheard excuse"));
        }

        private String setupPrompt() {
            return "Start a new case for Dead to Rights. Return exactly two tagged sections: <public> and <hidden_truth>.\n\n"
                    + "Inside <public>, print only CASE TITLE, SUSPECT, SETTING, CLUES 1-3, and 'You may begin questioning me.'\n"
                    + "Inside <hidden_truth>, write the actual crime, motive, method, mistakes, why each clue points to you, and the confession you would give if caught.\n"
                    + "Do not put hidden truth outside <hidden_truth>. Do not omit either tag.\n\n"
                    + "Freshness token: " + id + "\n"
                    + "Use this variation and do not reuse prior cases:\n"
                    + "- setting: " + setting + "\n"
                    + "- nonviolent crime type: " + crime + "\n"
                    + "- important object: " + object + "\n"
                    + "- suspect role you are playing: " + suspectRole + "\n"
                    + "- one clue style: " + clueStyle + "\n"
                    + "Current time: " + Instant.now();
        }

        private String shortLabel() {
            return crime + " at " + setting;
        }

        private static String pick(Random random, String... values) {
            return values[random.nextInt(values.length)];
        }
    }

    private record Options(String baseUrl, String model, int maxTokens, BigDecimal temperature, boolean help) {
        static Options parse(String[] args) {
            String baseUrl = "http://localhost:8085";
            String model = "Qwen3-4B-JQ4";
            int maxTokens = 1024;
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
            System.out.println("  --base-url http://localhost:18087");
            System.out.println("  --model Qwen3-4B-JQ4");
            System.out.println("  --max-tokens 1024");
            System.out.println("  --temperature 0.8");
        }
    }
}
