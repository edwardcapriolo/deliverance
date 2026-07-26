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
    private static final int CASE_CONTEXT_MESSAGES = 3;
    private static final int RECENT_INTERROGATION_MESSAGES = 4;
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
    private String publicCase = "";
    private String hiddenTruth = "";

    private DeadToRightsGame(Options options) {
        this.options = options;
        this.seed = (int) (System.currentTimeMillis() & 0x7fffffff);
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
        printBanner();
        Random random = new Random((((long) seed) << 32) ^ System.nanoTime());
        String suspectName = generateSuspectName();
        System.out.println(BOLD + CYAN + "The suspect's name is " + RESET + suspectName);

        List<String> places = generateChoices("places", "Generate 10 ordinary present-day locations for a light non-violent mystery game. Use grounded places like a bakery, radio station, community theater, office, bookshop, school fundraiser, museum office, hotel desk, garden center, or local club. Avoid fantasy, supernatural, hidden temples, magic objects, ancient rituals, whispering voices, changing books, glowing symbols, locked-room melodrama, remote cabins, spooky mansions, and exotic mystery settings.");
        String place = pickAndPrint("places", places, random);

        List<String> items = generateChoices("items", "Generate 10 ordinary non-magical objects that could be stolen in a light non-violent mystery game. Use grounded items like a trophy, ledger, donation envelope, recipe card, signed book, office key, antique vase, raffle tickets, camera, or cash box. Avoid magical, supernatural, glowing, future-predicting, cursed, enchanted, or fantasy items.");
        String item = pickAndPrint("items", items, random);

        List<Map<String, String>> messages = new ArrayList<>();
        messages.add(message("system", systemPrompt()));
        messages.add(message("user", caseSetupPrompt(suspectName, place, item)));

        System.out.println(DIM + "A crime has been committed. The investigating officer has brought in the suspect "
                + "and is preparing the details for you. The suspect is waiting in the interrogation room. "
                + "You are our best interrogator; we need you to get in there and get a confession!" + RESET);
        String setup = chat(messages, false, caseFileSchema());
        if (setup.isBlank()) {
            throw new IOException("The model produced no visible case opening. Try a larger --max-tokens value or type /showthink on the next run to inspect hidden reasoning.");
        }
        CaseFile caseFile = parseCaseFile(setup);
        publicCase = caseFile.publicOpening();
        hiddenTruth = caseFile.hiddenReveal();
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
                    continue;
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
                String response = chat(messages);
                messages.add(message("assistant", response));
                if (confessed(response)) {
                    System.out.println();
                    System.out.println(BOLD + GREEN + "DEAD TO RIGHTS" + RESET + " | the suspect confessed");
                    return;
                }
                trimConversation(messages);
            }
        }
    }

    private static void trimConversation(List<Map<String, String>> messages) {
        int keep = CASE_CONTEXT_MESSAGES + RECENT_INTERROGATION_MESSAGES;
        if (messages.size() <= keep) {
            return;
        }
        List<Map<String, String>> compact = new ArrayList<>();
        compact.addAll(messages.subList(0, CASE_CONTEXT_MESSAGES));
        compact.addAll(messages.subList(messages.size() - RECENT_INTERROGATION_MESSAGES, messages.size()));
        messages.clear();
        messages.addAll(compact);
    }

    static String systemPrompt() {
        return "You are running a fictional interrogation mystery game called Dead to Rights."
                + "Create a fictional crime scenario. You are the culprit. "
                + "You are the guilty suspect being questioned by the user."
                + "The crime must be non-violent, such as theft."
                + "Secretly decide what crime you committed, your motive, how you did it, what mistakes you made, and three clues that point toward you."
                + "At the start, reveal only CASE TITLE, SUSPECT, SETTING, and exactly three CLUES. Do not reveal the hidden truth."
                + "During play, answer every user question in first person as the guilty suspect. Pretend to be innocent, but keep the mystery fun: give specific, useful details, partial truths, suspicious excuses, so the user can follow up on. If asked about an object, place, person, time, or motive, invent a concrete answer that fits the hidden truth while still trying to deflect blame. "
                + "If the user presents a good theory that connects you to the crime using clues or contradictions, break down and confess. "
                + "When you confess, admit guilt: 'I confess. I took it because...'. "
                + "Explain what you did. Only then include exactly this marker: <confession>true</confession>.";

    }

    static boolean confessed(String text) {
        return text != null && text.toLowerCase().contains("<confession>true</confession>");
    }

    private String generateSuspectName() throws IOException {
        List<Map<String, String>> messages = List.of(
                message("system", "You create concise ordinary fictional names for a light mystery game. Avoid fantasy, gothic, supernatural, or melodramatic names."),
                message("user", "Generate only one fictional suspect name. No title, no explanation."));
        return stripNoise(chat(messages, false, null, BigDecimal.valueOf(1.3))).replaceAll("[\r\n]+", " ").strip();
    }

    private List<String> generateChoices(String fieldName, String prompt) throws IOException {
        List<Map<String, String>> messages = List.of(
                message("system", "You generate varied setup options for a light, non-violent mystery game."),
                message("user", prompt + " Return only JSON matching the schema."));
        JsonNode node = JSON.readTree(chat(messages, true, choicesSchema(fieldName), BigDecimal.valueOf(1.2)));
        JsonNode array = node.path(fieldName);
        List<String> values = new ArrayList<>();
        for (JsonNode value : array) {
            if (value.isTextual() && !value.asText().isBlank()) {
                values.add(value.asText().strip());
            }
        }
        if (values.size() < 10) {
            throw new IOException("Expected 10 " + fieldName + ", got " + values + " from " + node);
        }
        return values.subList(0, 10);
    }

    private String pickAndPrint(String label, List<String> values, Random random) {
        System.out.println(BOLD + CYAN + label + RESET);
        for (int i = 0; i < values.size(); i++) {
            System.out.println((i + 1) + ". " + values.get(i));
        }
        String selected = values.get(random.nextInt(values.size()));
        System.out.println(BOLD + GREEN + "selected " + label.substring(0, label.length() - 1) + ": " + RESET + selected);
        System.out.println();
        return selected;
    }

    private static String stripNoise(String text) {
        return text == null ? "" : text.replace("\"", "").replace("`", "").strip();
    }

    private String chat(List<Map<String, String>> messages) throws IOException {
        return chat(messages, true, null, options.temperature);
    }

    private String chat(List<Map<String, String>> messages, boolean printOutput, Map<String, Object> guidedJson) throws IOException {
        return chat(messages, printOutput, guidedJson, options.temperature);
    }

    private String chat(List<Map<String, String>> messages, boolean printOutput, Map<String, Object> guidedJson,
            BigDecimal temperature) throws IOException {
        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model(options.model)
                .maxTokens(options.maxTokens)
                .seed(seed)
                .temperature(temperature)
                .xtcThreshold(options.xtcThreshold)
                .xtcProbability(options.xtcProbability)
                .stream(true)
                .chatTemplateKwargs(Map.of("enable_thinking", false))
                .messages((List) messages);
        if (guidedJson != null) {
            request.guidedJson(guidedJson);
        }
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

    static CaseFile parseCaseFile(String json) throws IOException {
        CaseFile caseFile = JSON.readValue(json, CaseFile.class);
        if (caseFile.caseTitle == null || caseFile.caseTitle.isBlank()
                || caseFile.suspect == null || caseFile.suspect.isBlank()
                || caseFile.setting == null || caseFile.setting.isBlank()
                || caseFile.crimeDescription == null || caseFile.crimeDescription.isBlank()
                || caseFile.meansClue == null || caseFile.meansClue.isBlank()
                || caseFile.opportunityClue == null || caseFile.opportunityClue.isBlank()
                || caseFile.mistakeClue == null || caseFile.mistakeClue.isBlank()
                || caseFile.hiddenTruth == null) {
            throw new IOException("Invalid Dead to Rights case JSON: " + json);
        }
        return caseFile;
    }

    static Map<String, Object> caseFileSchema() {
        Map<String, Object> string = Map.of("type", "string");
        return Map.of(
                "type", "object",
                "additionalProperties", false,
                "required", List.of("caseTitle", "suspect", "setting", "crimeDescription", "meansClue",
                        "opportunityClue", "mistakeClue", "hiddenTruth"),
                "properties", Map.of(
                        "caseTitle", string,
                        "suspect", string,
                        "setting", string,
                        "crimeDescription", string,
                        "meansClue", string,
                        "opportunityClue", string,
                        "mistakeClue", string,
                        "hiddenTruth", Map.of(
                                "type", "object",
                                "additionalProperties", false,
                                "required", List.of("crime", "method", "mistakes", "whyCluesMatter"),
                                "properties", Map.of(
                                        "crime", string,
                                        "method", string,
                                        "mistakes", Map.of("type", "array", "items", string),
                                        "whyCluesMatter", Map.of("type", "array", "items", string)))));
    }

    static Map<String, Object> choicesSchema(String fieldName) {
        return Map.of(
                "type", "object",
                "additionalProperties", false,
                "required", List.of(fieldName),
                "properties", Map.of(
                        fieldName, Map.of("type", "array", "items", Map.of("type", "string"))));
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
        System.out.println(CYAN + "│ " + RESET + BOLD + "Interrogate the suspect. Find the lie." + RESET + CYAN + "         │" + RESET);
        System.out.println(CYAN + "╰────────────────────────────────────────────────╯" + RESET);
        System.out.println(YELLOW + "No one gets hurt. Catch the culprit in a light fictional mystery." + RESET);
        System.out.println(DIM + "Commands: /case, /reveal, /quit, /q" + RESET);
        System.out.println(DIM + "model: " + options.model + " @ " + options.baseUrl + " | seed=" + seed + RESET);
        System.out.println();
    }

    private String caseSetupPrompt(String suspectName, String place, String item) {
        return "Start a new case for Dead to Rights.\n"
                + "The suspect is: " + suspectName + "\n"
                + "The setting is: " + place + "\n"
                + "The suspect stole this item: " + item + "\n"
                    + "The public fields are caseTitle, suspect, setting, crimeDescription, meansClue, opportunityClue, and mistakeClue.\n"
                    + "crimeDescription must be one simple sentence that states exactly what was stolen, for example: 'The cash box contents were stolen.' Do not confuse a container with its contents.\n"
                    + "The hiddenTruth object contains the actual crime, method, mistakes, and why the clues matter.\n"
                    + "A clue must be a concrete observable fact, such as a receipt, timestamp, key, witness statement, misplaced object, log entry, footprint, note, damaged lock, altered record, or contradiction.\n"
                    + "Do not use the suspect name, setting, or stolen item alone as a clue.\n"
                    + "meansClue must show how the suspect could access or take the item.\n"
                    + "opportunityClue must place the suspect near the item at the relevant time.\n"
                    + "mistakeClue must show an error the suspect made while hiding or covering up the theft.\n"
                + "The crime is theft. The theft went wrong in three ways, creating the three clues.\n"
                + "The suspect either left evidence, was seen by a witness, contradicted a timeline, had unusual access, or hid the item poorly.\n"
                + "Freshness token: " + UUID.randomUUID() + "\n"
                + "Current time: " + Instant.now();
    }

    static final class CaseFile {
        public String caseTitle;
        public String suspect;
        public String setting;
        public String crimeDescription;
        public String meansClue;
        public String opportunityClue;
        public String mistakeClue;
        public HiddenTruth hiddenTruth;

        String publicOpening() {
            StringBuilder sb = new StringBuilder();
            sb.append("CASE TITLE: ").append(caseTitle).append('\n');
            sb.append("SUSPECT: ").append(suspect).append('\n');
            sb.append("SETTING: ").append(setting).append('\n');
            sb.append("CRIME: ").append(crimeDescription).append('\n');
            sb.append("CLUES:\n");
            sb.append("1. ").append(meansClue).append('\n');
            sb.append("2. ").append(opportunityClue).append('\n');
            sb.append("3. ").append(mistakeClue).append('\n');
            sb.append("\nYou may begin questioning me.");
            return sb.toString();
        }

        String hiddenReveal() {
            return "CRIME: " + hiddenTruth.crime + "\n"
                    + "METHOD: " + hiddenTruth.method + "\n"
                    + "MISTAKES: " + hiddenTruth.mistakes + "\n"
                    + "WHY THE CLUES MATTER: " + hiddenTruth.whyCluesMatter;
        }
    }

    static final class HiddenTruth {
        public String crime;
        public String method;
        public List<String> mistakes;
        public List<String> whyCluesMatter;
    }

    private record Options(String baseUrl, String model, int maxTokens, BigDecimal temperature,
            BigDecimal xtcThreshold, BigDecimal xtcProbability, boolean help) {
        static Options parse(String[] args) {
            String baseUrl = "http://localhost:8085";
            String model = "Qwen3-4B-JQ4";
            int maxTokens = 1024;
            BigDecimal temperature = BigDecimal.valueOf(0.8);
            BigDecimal xtcThreshold = BigDecimal.valueOf(0.1);
            BigDecimal xtcProbability = BigDecimal.valueOf(0.2);
            boolean help = false;
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--help", "-h" -> help = true;
                    case "--base-url" -> baseUrl = args[++i];
                    case "--model" -> model = args[++i];
                    case "--max-tokens" -> maxTokens = Integer.parseInt(args[++i]);
                    case "--temperature" -> temperature = new BigDecimal(args[++i]);
                    case "--xtc-threshold" -> xtcThreshold = new BigDecimal(args[++i]);
                    case "--xtc-probability" -> xtcProbability = new BigDecimal(args[++i]);
                    default -> throw new IllegalArgumentException("Unknown argument: " + args[i]);
                }
            }
            return new Options(baseUrl.replaceAll("/+$", ""), model, maxTokens, temperature, xtcThreshold,
                    xtcProbability, help);
        }

        static void printHelp() {
            System.out.println("Dead to Rights options:");
            System.out.println("  --base-url http://localhost:18087");
            System.out.println("  --model Qwen3-4B-JQ4");
            System.out.println("  --max-tokens 1024");
            System.out.println("  --temperature 0.8");
            System.out.println("  --xtc-threshold 0.1");
            System.out.println("  --xtc-probability 0.2");
        }
    }
}
