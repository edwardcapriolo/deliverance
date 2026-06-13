package io.teknek.deliverance.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.DefaultCausalLanguageModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.CausalLanguageModel;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.model.tensorparallel.GossipParallelMembership;
import io.teknek.deliverance.model.tensorparallel.GossipParallelSettings;
import io.teknek.deliverance.model.tensorparallel.TensorParallelDeploymentSpec;
import io.teknek.deliverance.model.tensorparallel.TensorParallelGenerationGroup;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.gossip.GossipSettings;
import io.teknek.gossip.Member;
import io.teknek.gossip.RemoteMember;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.UUID;
import java.util.function.BooleanSupplier;

/**
 * Command-line inference throughput benchmark for Deliverance and Ollama.
 *
 * <p>This is intentionally a main program, not a unit test. It runs a prompt suite with long, mixed-category requests
 * and writes per-turn timing rows that can be compared across engines. The built-in suite is a small MT-Bench-derived
 * subset covering reasoning, math, coding, extraction, STEM, humanities, and writing. For broader coverage, pass the
 * FastChat MT-Bench {@code question.jsonl} file with {@code --suite-file}.</p>
 *
 * <p>Ollama timing is read from its final {@code /api/chat} response fields: {@code prompt_eval_count},
 * {@code eval_count}, {@code total_duration}, and {@code eval_duration}. Deliverance timing comes from
 * {@link Response#promptTokens}, {@link Response#generatedTokens}, and {@link Response#totalTimeMs}.</p>
 */
public final class InferenceBenchmark {
    private static final String CSV_HEADER = String.join(",",
            "timestamp",
            "engine",
            "model",
            "case_id",
            "category",
            "turn",
            "prompt_chars",
            "prompt_tokens",
            "generated_tokens",
            "total_ms",
            "generation_ms",
            "tokens_per_second",
            "response_chars",
            "finish_reason");

    private InferenceBenchmark() {
    }

    /**
     * Runs the benchmark.
     *
     * <p>Common examples:</p>
     *
     * <pre>{@code
     * java ... io.teknek.deliverance.benchmark.InferenceBenchmark \
     *   --engine deliverance --owner tjake --model gemma-2-2b-it-JQ4 --output deliverance.csv
     *
     * java ... io.teknek.deliverance.benchmark.InferenceBenchmark \
     *   --engine ollama --ollama-model llama3.2 --output ollama.csv
     *
     * java ... io.teknek.deliverance.benchmark.InferenceBenchmark \
     *   --engine both --owner tjake --model gemma-2-2b-it-JQ4 --ollama-model llama3.2
     * }</pre>
     */
    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        List<BenchmarkCase> cases = options.suiteFile == null ? builtInSuite() : loadMtBenchJsonl(options.suiteFile);
        if (options.maxCases > 0 && options.maxCases < cases.size()) {
            cases = cases.subList(0, options.maxCases);
        }
        if (options.output.getParent() != null) {
            Files.createDirectories(options.output.getParent());
        }
        if (options.jsonlOutput != null && options.jsonlOutput.getParent() != null) {
            Files.createDirectories(options.jsonlOutput.getParent());
        }
        try (BufferedWriter writer = Files.newBufferedWriter(options.output, StandardCharsets.UTF_8);
             BufferedWriter jsonlWriter = options.jsonlOutput == null ? null
                     : Files.newBufferedWriter(options.jsonlOutput, StandardCharsets.UTF_8)) {
            writer.write(CSV_HEADER);
            writer.newLine();
            writer.flush();
            System.out.println("writing benchmark results to " + options.output.toAbsolutePath());
            if (jsonlWriter != null) {
                System.out.println("writing benchmark transcripts to " + options.jsonlOutput.toAbsolutePath());
            }
            if (options.engine == Engine.DELIVERANCE || options.engine == Engine.BOTH) {
                runDeliverance(options, cases, writer, jsonlWriter);
            }
            if (options.engine == Engine.OLLAMA || options.engine == Engine.BOTH) {
                runOllama(options, cases, writer, jsonlWriter);
            }
        }
        System.out.println("wrote benchmark results to " + options.output.toAbsolutePath());
    }

    /**
     * Runs the prompt suite against an embedded Deliverance model.
     *
     * <p>Each benchmark case is a conversation. For multi-turn cases, previous model responses are added back as
     * assistant messages before the next user turn, matching MT-Bench style evaluation.</p>
     */
    private static void runDeliverance(Options options, List<BenchmarkCase> cases, BufferedWriter writer,
            BufferedWriter jsonlWriter) throws IOException {
        ModelFetcher fetcher = new ModelFetcher(options.owner, options.model);
        System.out.println("[deliverance] loading model " + options.owner + "/" + options.model
                + " tensor_parallel_size=" + options.tensorParallelSize);
        try (DeliveranceRunner runner = openDeliveranceRunner(options, fetcher)) {
            System.out.println("[deliverance] loaded model " + runner.modelName()
                    + "; cases=" + cases.size() + "; warmup_cases=" + options.warmupCases);
            runner.printRuntime();
            for (int warmup = 0; warmup < options.warmupCases && warmup < cases.size(); warmup++) {
                runDeliveranceCase(runner, options, cases.get(warmup), null, null, true);
            }
            for (BenchmarkCase benchmarkCase : cases) {
                runDeliveranceCase(runner, options, benchmarkCase, writer, jsonlWriter, false);
                writer.flush();
            }
        }
    }

    /** Opens either a single-model or local in-process tensor-parallel Deliverance runner. */
    private static DeliveranceRunner openDeliveranceRunner(Options options, ModelFetcher fetcher) {
        if (options.tensorParallelSize <= 1) {
            return new LocalDeliveranceRunner(options.owner + "/" + options.model,
                    AutoModelForCausaLm.newBuilder(fetcher)
                    .withWorkingMemoryType(options.workingDType)
                    .withWorkingQuantType(options.workingQType)
                    .build());
        }
        return GossipTensorParallelDeliveranceRunner.open(options, fetcher);
    }

    /** Prints runtime details that are otherwise only visible through SLF4J logs. */
    private static void printDeliveranceRuntime(CausalLanguageModel model) {
        if (model instanceof DefaultCausalLanguageModel defaultModel) {
            printLocalModelRuntime("coordinator", defaultModel.localTransformerModel());
        } else {
            System.out.println("[deliverance] runtime_details_unavailable implementation="
                    + model.getClass().getName());
        }
    }

    /** Prints runtime details for one local transformer executor. */
    private static void printLocalModelRuntime(String label, AbstractModel localModel) {
            System.out.println("[deliverance] " + label
                    + " tensor_provider=" + localModel.getTensorProviderName()
                    + " parallel_split_size=" + localModel.getTensorProviderParallelSplitSize()
                    + " model_dtype=" + localModel.getModelDType()
                    + " working_dtype=" + localModel.getWorkingDType()
                    + " working_qtype=" + localModel.getWorkingQType()
                    + " tp_rank=" + localModel.getTensorParallelContext().rank()
                    + " tp_size=" + localModel.getTensorParallelContext().size());
    }

    /** Runs one Deliverance conversation and optionally writes per-turn result rows. */
    private static void runDeliveranceCase(DeliveranceRunner runner, Options options, BenchmarkCase benchmarkCase,
            BufferedWriter writer, BufferedWriter jsonlWriter, boolean warmup) throws IOException {
        List<ChatMessage> messages = new ArrayList<>();
        Optional<PromptSupport> promptSupport = runner.promptSupport();
        for (int turn = 0; turn < benchmarkCase.turns.size(); turn++) {
            messages.add(new ChatMessage("user", benchmarkCase.turns.get(turn)));
            PromptContext promptContext = promptContext(promptSupport, messages);
            printStart("deliverance", runner.modelName(), benchmarkCase, turn + 1, warmup,
                    promptContext.getPrompt().length());
            GeneratorParameters parameters = new GeneratorParameters()
                    .withTemperature(options.temperature)
                    .withMaxTokens(options.maxTokens);
            if (options.seed != null) {
                parameters.withSeed(options.seed);
            }
            Response response = runner.generate(UUID.randomUUID(), promptContext, parameters);
            messages.add(new ChatMessage("assistant", response.responseText));
            if (!warmup) {
                double generationMs = Math.max(0.0, response.totalTimeMs - response.timeToFirstTokenMs);
                long decodeTokens = Math.max(0, response.generatedTokens.size() - 1L);
                double tokensPerSecond = generationMs == 0.0
                        ? 0.0
                        : decodeTokens / (generationMs / 1000.0);
                ResultRow row = new ResultRow(
                        "deliverance",
                        runner.modelName(),
                        benchmarkCase.id,
                        benchmarkCase.category,
                        turn + 1,
                        promptContext.getPrompt().length(),
                        response.promptTokens,
                        response.generatedTokens.size(),
                        response.totalTimeMs,
                        generationMs,
                        tokensPerSecond,
                        response.responseText.length(),
                        response.finishReason == null ? "" : response.finishReason.name());
                writeRow(writer, row);
                writer.flush();
                writeTranscript(jsonlWriter, row, messages, promptContext.getPrompt(), response.responseText,
                        response.responseTextWithSpecialTokens);
                printProgress("deliverance", runner.modelName(), benchmarkCase, turn + 1,
                        response.promptTokens, response.generatedTokens.size(), response.totalTimeMs, tokensPerSecond,
                        response.finishReason == null ? "" : response.finishReason.name());
            } else {
                System.out.printf(Locale.ROOT,
                        "[deliverance] warmup complete case=%s category=%s turn=%d generated=%d total_ms=%.1f%n",
                        benchmarkCase.id,
                        benchmarkCase.category,
                        turn + 1,
                        response.generatedTokens.size(),
                        response.totalTimeMs);
            }
        }
    }

    /** Common generation surface used by single-model and local tensor-parallel benchmark modes. */
    private interface DeliveranceRunner extends AutoCloseable {
        String modelName();

        Optional<PromptSupport> promptSupport();

        Response generate(UUID sessionId, PromptContext promptContext, GeneratorParameters parameters);

        void printRuntime();

        @Override
        void close();
    }

    /** Single local-model benchmark runner. */
    private static final class LocalDeliveranceRunner implements DeliveranceRunner {
        private final String modelName;
        private final CausalLanguageModel model;

        private LocalDeliveranceRunner(String modelName, CausalLanguageModel model) {
            this.modelName = modelName;
            this.model = model;
        }

        @Override
        public String modelName() {
            return modelName;
        }

        @Override
        public Optional<PromptSupport> promptSupport() {
            return model.promptSupport();
        }

        @Override
        public Response generate(UUID sessionId, PromptContext promptContext, GeneratorParameters parameters) {
            return model.generate(sessionId, promptContext, parameters, new DoNothingGenerateEvent());
        }

        @Override
        public void printRuntime() {
            printDeliveranceRuntime(model);
        }

        @Override
        public void close() {
            try {
                model.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    /** Gossip/worker tensor-parallel benchmark runner that mirrors {@code Gemma2TensorParallelIT}. */
    private static final class GossipTensorParallelDeliveranceRunner implements DeliveranceRunner {
        private final String modelName;
        private final AbstractModel coordinatorModel;
        private final TensorParallelGenerationGroup group;
        private final List<BenchmarkNode> nodes;

        private GossipTensorParallelDeliveranceRunner(String modelName, AbstractModel coordinatorModel,
                TensorParallelGenerationGroup group, List<BenchmarkNode> nodes) {
            this.modelName = modelName;
            this.coordinatorModel = coordinatorModel;
            this.group = group;
            this.nodes = nodes;
        }

        private static GossipTensorParallelDeliveranceRunner open(Options options, ModelFetcher fetcher) {
            if (options.tensorParallelSize < 2) {
                throw new IllegalArgumentException("tensorParallelSize must be >= 2");
            }
            String cluster = "deliverance-benchmark-tp-" + UUID.randomUUID();
            int basePort = 42_000 + Math.floorMod(cluster.hashCode(), 1_000);
            String node0 = "node-0";
            String node1 = "node-1";
            URI node0Uri = URI.create("udp://127.0.0.1:" + basePort);
            URI node1Uri = URI.create("udp://127.0.0.1:" + (basePort + 1));
            List<Member> seedMembers = List.of(new RemoteMember(cluster, node0Uri, node0),
                    new RemoteMember(cluster, node1Uri, node1));
            GossipSettings settings = new GossipSettings();
            settings.setPersistRingState(false);
            settings.setPersistDataState(false);
            settings.setGossipInterval(100);
            settings.setCleanupInterval(2_000);
            TensorParallelDeploymentSpec deploymentSpec = new TensorParallelDeploymentSpec("benchmark",
                    options.tensorParallelSize, options.tensorParallelMaxRanksPerWorker);
            List<BenchmarkNode> nodes = new ArrayList<>();
            try {
                nodes.add(createNode(options, fetcher, cluster, node0, node0Uri, seedMembers, settings, deploymentSpec));
                nodes.add(createNode(options, fetcher, cluster, node1, node1Uri, seedMembers, settings, deploymentSpec));
                eventually(() -> allMembersVisible(nodes), Duration.ofSeconds(10));
                eventually(() -> allCandidatesVisible(nodes, deploymentSpec.minimumPhysicalNodes()), Duration.ofSeconds(10));
                eventually(() -> allNodesSeeLeader(nodes, node0), Duration.ofSeconds(10));
                eventually(() -> allNodesSeeAssignment(nodes), Duration.ofSeconds(10));
                eventually(() -> allNodesSeeCollectiveUri(nodes), Duration.ofSeconds(10));
                eventually(() -> allNodesSeeRankEndpoints(nodes), Duration.ofSeconds(10));
                TensorParallelGenerationGroup group = nodes.getFirst().membership().openGenerationGroup();
                AbstractModel coordinator = AutoModelForCausaLm.newBuilder(fetcher)
                        .withWorkingMemoryType(options.workingDType)
                        .withWorkingQuantType(options.workingQType)
                        .buildLocalTransformerModel();
                return new GossipTensorParallelDeliveranceRunner(options.owner + "/" + options.model
                        + ":tp" + options.tensorParallelSize + "x" + nodes.size(), coordinator, group, List.copyOf(nodes));
            } catch (RuntimeException e) {
                nodes.forEach(BenchmarkNode::close);
                throw e;
            }
        }

        private static BenchmarkNode createNode(Options options, ModelFetcher fetcher, String cluster, String nodeId,
                URI nodeUri, List<Member> seedMembers, GossipSettings settings,
                TensorParallelDeploymentSpec deploymentSpec) {
            AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher)
                    .withWorkingMemoryType(options.workingDType)
                    .withWorkingQuantType(options.workingQType)
                    .withParallelSettings(new GossipParallelSettings(cluster, nodeId, nodeUri, seedMembers, settings,
                            deploymentSpec))
                    .buildAbstractModel();
            return new BenchmarkNode(nodeId, model, model.gossipParallelMembership().orElseThrow());
        }

        private static boolean allMembersVisible(List<BenchmarkNode> nodes) {
            return nodes.stream().allMatch(node -> node.membership().liveMembers().size() == nodes.size() - 1);
        }

        private static boolean allCandidatesVisible(List<BenchmarkNode> nodes, int expectedCandidates) {
            return nodes.stream().allMatch(node -> node.membership().candidateNodeIds().size() == expectedCandidates);
        }

        private static boolean allNodesSeeLeader(List<BenchmarkNode> nodes, String leaderNodeId) {
            return nodes.stream().allMatch(node -> leaderNodeId.equals(node.membership().electedLeader()));
        }

        private static boolean allNodesSeeAssignment(List<BenchmarkNode> nodes) {
            return nodes.stream().allMatch(node -> node.membership().findAssignment() != null);
        }

        private static boolean allNodesSeeCollectiveUri(List<BenchmarkNode> nodes) {
            return nodes.stream().allMatch(node -> node.membership().findCollectiveUri() != null);
        }

        private static boolean allNodesSeeRankEndpoints(List<BenchmarkNode> nodes) {
            return nodes.stream().allMatch(observer -> nodes.stream()
                    .allMatch(owner -> observer.membership().findRankEndpoints(owner.id()).size()
                            == owner.membership().localRanks().size()));
        }

        private static void eventually(BooleanSupplier condition, Duration timeout) {
            long deadline = System.nanoTime() + timeout.toNanos();
            while (System.nanoTime() < deadline) {
                if (condition.getAsBoolean()) {
                    return;
                }
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("interrupted waiting for tensor-parallel benchmark setup", e);
                }
            }
            throw new IllegalStateException("tensor-parallel benchmark setup did not become ready within " + timeout);
        }

        @Override
        public String modelName() {
            return modelName;
        }

        @Override
        public Optional<PromptSupport> promptSupport() {
            return coordinatorModel.promptSupport();
        }

        @Override
        public Response generate(UUID sessionId, PromptContext promptContext, GeneratorParameters parameters) {
            return group.generate(sessionId, coordinatorModel, promptContext, parameters, new DoNothingGenerateEvent());
        }

        @Override
        public void printRuntime() {
            printLocalModelRuntime("coordinator", coordinatorModel);
            for (BenchmarkNode node : nodes) {
                printLocalModelRuntime(node.id(), node.model());
            }
        }

        @Override
        public void close() {
            group.close();
            coordinatorModel.close();
            nodes.forEach(BenchmarkNode::close);
        }
    }

    private record BenchmarkNode(String id, AbstractModel model, GossipParallelMembership membership) implements AutoCloseable {
        @Override
        public void close() {
            model.close();
        }
    }

    /**
     * Runs the prompt suite against Ollama's {@code /api/chat} endpoint.
     *
     * <p>Ollama must already be running and the selected model must be pulled. This runner sends non-streaming chat
     * requests so the final timing counters are available in one JSON response.</p>
     */
    private static void runOllama(Options options, List<BenchmarkCase> cases, BufferedWriter writer, BufferedWriter jsonlWriter)
            throws IOException, InterruptedException {
        HttpClient client = HttpClient.newHttpClient();
        System.out.println("[ollama] using model " + options.ollamaModel + " at " + options.ollamaBaseUrl
                + "; cases=" + cases.size() + "; warmup_cases=" + options.warmupCases);
        for (int warmup = 0; warmup < options.warmupCases && warmup < cases.size(); warmup++) {
            runOllamaCase(client, options, cases.get(warmup), null, null, true);
        }
        for (BenchmarkCase benchmarkCase : cases) {
            runOllamaCase(client, options, benchmarkCase, writer, jsonlWriter, false);
            writer.flush();
        }
    }

    /** Runs one Ollama conversation and optionally writes per-turn result rows. */
    private static void runOllamaCase(HttpClient client, Options options, BenchmarkCase benchmarkCase, BufferedWriter writer,
            BufferedWriter jsonlWriter, boolean warmup) throws IOException, InterruptedException {
        List<ChatMessage> messages = new ArrayList<>();
        for (int turn = 0; turn < benchmarkCase.turns.size(); turn++) {
            messages.add(new ChatMessage("user", benchmarkCase.turns.get(turn)));
            ObjectNode requestJson = JsonUtils.om.createObjectNode();
            requestJson.put("model", options.ollamaModel);
            requestJson.put("stream", false);
            ObjectNode optionsJson = JsonUtils.om.createObjectNode();
            optionsJson.put("temperature", options.temperature);
            optionsJson.put("num_predict", options.maxTokens);
            if (options.seed != null) {
                optionsJson.put("seed", options.seed);
            }
            requestJson.set("options", optionsJson);
            ArrayNode messagesJson = requestJson.putArray("messages");
            for (ChatMessage message : messages) {
                ObjectNode node = messagesJson.addObject();
                node.put("role", message.role);
                node.put("content", message.content);
            }
            printStart("ollama", options.ollamaModel, benchmarkCase, turn + 1, warmup, requestJson.toString().length());
            HttpRequest request = HttpRequest.newBuilder(options.ollamaBaseUrl.resolve("/api/chat"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(JsonUtils.om.writeValueAsString(requestJson)))
                    .build();
            HttpResponse<String> httpResponse = client.send(request, HttpResponse.BodyHandlers.ofString());
            if (httpResponse.statusCode() / 100 != 2) {
                throw new IOException("Ollama request failed status=" + httpResponse.statusCode() + " body="
                        + httpResponse.body());
            }
            JsonNode response = JsonUtils.om.readTree(httpResponse.body());
            String assistant = response.path("message").path("content").asText("");
            messages.add(new ChatMessage("assistant", assistant));
            if (!warmup) {
                long evalCount = response.path("eval_count").asLong(0);
                long evalDurationNanos = response.path("eval_duration").asLong(0);
                double generationMs = evalDurationNanos / 1_000_000.0;
                double tokensPerSecond = evalDurationNanos == 0 ? 0.0 : evalCount / (evalDurationNanos / 1_000_000_000.0);
                ResultRow row = new ResultRow(
                        "ollama",
                        options.ollamaModel,
                        benchmarkCase.id,
                        benchmarkCase.category,
                        turn + 1,
                        requestJson.toString().length(),
                        response.path("prompt_eval_count").asLong(0),
                        evalCount,
                        response.path("total_duration").asDouble(0.0) / 1_000_000.0,
                        generationMs,
                        tokensPerSecond,
                        assistant.length(),
                        response.path("done_reason").asText(""));
                writeRow(writer, row);
                writer.flush();
                writeTranscript(jsonlWriter, row, messages, JsonUtils.om.writeValueAsString(requestJson), assistant, assistant);
                printProgress("ollama", options.ollamaModel, benchmarkCase, turn + 1,
                        response.path("prompt_eval_count").asLong(0), evalCount,
                        response.path("total_duration").asDouble(0.0) / 1_000_000.0, tokensPerSecond,
                        response.path("done_reason").asText(""));
            } else {
                System.out.printf(Locale.ROOT,
                        "[ollama] warmup complete case=%s category=%s turn=%d generated=%d total_ms=%.1f%n",
                        benchmarkCase.id,
                        benchmarkCase.category,
                        turn + 1,
                        response.path("eval_count").asLong(0),
                        response.path("total_duration").asDouble(0.0) / 1_000_000.0);
            }
        }
    }

    /** Prints a progress line before a potentially long model invocation starts. */
    private static void printStart(String engine, String model, BenchmarkCase benchmarkCase, int turn, boolean warmup,
            int promptChars) {
        System.out.printf(Locale.ROOT,
                "[%s] %sstart model=%s case=%s category=%s turn=%d prompt_chars=%d%n",
                engine,
                warmup ? "warmup " : "",
                model,
                benchmarkCase.id,
                benchmarkCase.category,
                turn,
                promptChars);
    }

    /** Prints one concise progress line after a recorded benchmark turn completes. */
    private static void printProgress(String engine, String model, BenchmarkCase benchmarkCase, int turn, long promptTokens,
            long generatedTokens, double totalMs, double tokensPerSecond, String finishReason) {
        System.out.printf(Locale.ROOT,
                "[%s] model=%s case=%s category=%s turn=%d prompt_tokens=%d generated=%d total_ms=%.1f tok_s=%.2f finish=%s%n",
                engine,
                model,
                benchmarkCase.id,
                benchmarkCase.category,
                turn,
                promptTokens,
                generatedTokens,
                totalMs,
                tokensPerSecond,
                finishReason);
    }

    /** Builds a prompt context from chat messages, falling back to a raw transcript if no template is available. */
    private static PromptContext promptContext(Optional<PromptSupport> promptSupport, List<ChatMessage> messages) {
        if (promptSupport.isPresent()) {
            PromptSupport.Builder builder = promptSupport.get().builder();
            for (ChatMessage message : messages) {
                switch (message.role) {
                    case "user" -> builder.addUserMessage(message.content);
                    case "assistant" -> builder.addAssistantMessage(message.content);
                    case "system" -> builder.addSystemMessage(message.content);
                    default -> throw new IllegalArgumentException("Unsupported role " + message.role);
                }
            }
            return builder.build();
        }
        StringBuilder raw = new StringBuilder();
        for (ChatMessage message : messages) {
            raw.append(message.role).append(": ").append(message.content).append('\n');
        }
        raw.append("assistant: ");
        return PromptContext.of(raw.toString());
    }

    /** Loads FastChat MT-Bench style JSONL with fields {@code question_id}, {@code category}, and {@code turns}. */
    private static List<BenchmarkCase> loadMtBenchJsonl(Path suiteFile) throws IOException {
        List<BenchmarkCase> cases = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(suiteFile, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                JsonNode node = JsonUtils.om.readTree(line);
                List<String> turns = new ArrayList<>();
                for (JsonNode turn : node.withArray("turns")) {
                    turns.add(turn.asText());
                }
                cases.add(new BenchmarkCase(node.path("question_id").asText("case-" + cases.size()),
                        node.path("category").asText("unknown"), turns));
            }
        }
        return cases;
    }

    /** Writes one benchmark result row as CSV. */
    private static void writeRow(BufferedWriter writer, ResultRow row) throws IOException {
        writer.write(String.join(",",
                csv(Instant.now().toString()),
                csv(row.engine),
                csv(row.model),
                csv(row.caseId),
                csv(row.category),
                Integer.toString(row.turn),
                Integer.toString(row.promptChars),
                Long.toString(row.promptTokens),
                Long.toString(row.generatedTokens),
                Double.toString(row.totalMs),
                Double.toString(row.generationMs),
                Double.toString(row.tokensPerSecond),
                Integer.toString(row.responseChars),
                csv(row.finishReason)));
        writer.newLine();
    }

    /** Writes one JSONL transcript record, if transcript output is enabled. */
    private static void writeTranscript(BufferedWriter jsonlWriter, ResultRow row, List<ChatMessage> messages,
            String renderedPrompt, String responseText, String responseTextWithSpecialTokens) throws IOException {
        if (jsonlWriter == null) {
            return;
        }
        ObjectNode root = JsonUtils.om.createObjectNode();
        root.put("timestamp", Instant.now().toString());
        root.put("engine", row.engine);
        root.put("model", row.model);
        root.put("case_id", row.caseId);
        root.put("category", row.category);
        root.put("turn", row.turn);
        ArrayNode messagesNode = root.putArray("messages");
        for (ChatMessage message : messages) {
            ObjectNode messageNode = messagesNode.addObject();
            messageNode.put("role", message.role);
            messageNode.put("content", message.content);
        }
        root.put("rendered_prompt", renderedPrompt);
        root.put("response", responseText);
        root.put("response_with_special_tokens", responseTextWithSpecialTokens);
        ObjectNode metrics = root.putObject("metrics");
        metrics.put("prompt_chars", row.promptChars);
        metrics.put("prompt_tokens", row.promptTokens);
        metrics.put("generated_tokens", row.generatedTokens);
        metrics.put("total_ms", row.totalMs);
        metrics.put("generation_ms", row.generationMs);
        metrics.put("tokens_per_second", row.tokensPerSecond);
        metrics.put("response_chars", row.responseChars);
        metrics.put("finish_reason", row.finishReason);
        jsonlWriter.write(JsonUtils.om.writeValueAsString(root));
        jsonlWriter.newLine();
        jsonlWriter.flush();
    }

    /** Escapes one CSV field. */
    private static String csv(String value) {
        String escaped = value == null ? "" : value.replace("\"", "\"\"");
        return "\"" + escaped + "\"";
    }

    /** Built-in mixed prompt suite derived from MT-Bench categories and task style. */
    private static List<BenchmarkCase> builtInSuite() {
        return List.of(
                new BenchmarkCase("builtin-reasoning-1", "reasoning", List.of(
                        "Read the puzzle carefully and answer with a clear explanation. A company reserves five parking spaces in order for the CEO, president, vice president, secretary, and treasurer. The cars are red, blue, green, yellow, and purple. The first space is red. A blue car is between the red car and the green car. The last space is purple. The secretary drives yellow. Alice parks next to David. Enid drives green. Bert parks between Cheryl and Enid. David parks in the last space. Who is the secretary, and what are the car colors from first to last?",
                        "Now explain which clues were necessary and which were redundant.")),
                new BenchmarkCase("builtin-math-1", "math", List.of(
                        "Solve step by step. A bus starts with an unknown number of passengers. At the first stop, half get off and 4 get on. At the second stop, 6 get off and 8 get on. There are 25 passengers heading to the third stop. How many passengers started at the terminal? Then compute total fare collected if every person who ever boarded paid $2.",
                        "Generalize the algebra for starting passengers S, first-stop additions A, second-stop exits B, second-stop additions C, and final passengers F.")),
                new BenchmarkCase("builtin-coding-1", "coding", List.of(
                        "Write a production-quality Python program that recursively scans a directory, reads all UTF-8 text files, ignores binary files, tokenizes words case-insensitively, and returns the top 10 words by frequency. Include error handling and a small explanation of complexity.",
                        "Parallelize the program safely. Explain when the parallel version may be slower than the sequential version.")),
                new BenchmarkCase("builtin-coding-2", "coding", List.of(
                        "Given two sorted arrays of different sizes, implement an O(log(min(m,n))) algorithm to find their median. Provide Python code, explain the partition invariant, and include edge cases for empty arrays and duplicates.",
                        "Now write five focused unit tests for the implementation and explain what bug each test would catch.")),
                new BenchmarkCase("builtin-extraction-1", "extraction", List.of(
                        "Extract the highest and lowest closing prices per month from this CSV and return compact JSON with month names as keys. Round to nearest integer.\nDate,Open,High,Low,Close,Volume\n2022-01-01,150.02,155.28,148.50,153.80,15678900\n2022-01-02,154.32,157.25,153.48,156.25,19874500\n2022-02-01,160.50,163.28,159.50,161.80,14326700\n2022-02-02,161.80,164.25,161.30,163.90,17689200\n2022-03-01,165.40,168.35,163.10,166.80,16253400\n2022-03-02,167.00,169.85,165.50,168.20,19568100",
                        "Change the JSON to CSV and include a third column for the spread between highest and lowest close.")),
                new BenchmarkCase("builtin-stem-1", "stem", List.of(
                        "Design a solar-powered water-heating system for a residential building that serves 100 people. Include components, sizing assumptions, five-step deployment workflow, rough budget, failure modes, and maintenance schedule.",
                        "Now identify the three biggest flaws in your design and quantify how each could affect cost or output.")),
                new BenchmarkCase("builtin-humanities-1", "humanities", List.of(
                        "Explain how GDP, inflation, and unemployment interact. Then compare how fiscal policy and monetary policy influence those indicators during a recession. Use concrete examples and caveats.",
                        "Explain the same ideas to a ten-year-old using an analogy, but preserve the important tradeoffs.")),
                new BenchmarkCase("builtin-writing-1", "writing", List.of(
                        "Draft a concise but persuasive memo to engineering leadership arguing for a monthly inference benchmark report. It should cover performance regressions, model quality drift, hardware differences, and user-facing latency.",
                        "Rewrite the memo as a checklist that a release manager can execute before every release.")));
    }

    private record BenchmarkCase(String id, String category, List<String> turns) {
    }

    private record ChatMessage(String role, String content) {
    }

    private record ResultRow(String engine, String model, String caseId, String category, int turn, int promptChars,
            long promptTokens, long generatedTokens, double totalMs, double generationMs, double tokensPerSecond,
            int responseChars, String finishReason) {
    }

    private enum Engine {
        DELIVERANCE,
        OLLAMA,
        BOTH
    }

    /** Parsed command-line options for the benchmark. */
    private static final class Options {
        private Engine engine = Engine.DELIVERANCE;
        private String owner = "tjake";
        private String model = "gemma-2-2b-it-JQ4";
        private String ollamaModel = "llama3.2";
        private URI ollamaBaseUrl = URI.create("http://localhost:11434");
        private DType workingDType = DType.F32;
        private DType workingQType = DType.I8;
        private int maxTokens = 256;
        private float temperature = 0.0f;
        private Integer seed = 42;
        private int warmupCases = 1;
        private int maxCases = 0;
        private int tensorParallelSize = 1;
        private int tensorParallelMaxRanksPerWorker = 2;
        private Path suiteFile;
        private Path output = Path.of("target/inference-benchmark.csv");
        private Path jsonlOutput;

        /** Parses {@code --key value} style command-line arguments. */
        private static Options parse(String[] args) {
            Options options = new Options();
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--engine" -> options.engine = Engine.valueOf(args[++i].toUpperCase(Locale.ROOT));
                    case "--owner" -> options.owner = args[++i];
                    case "--model" -> options.model = args[++i];
                    case "--ollama-model" -> options.ollamaModel = args[++i];
                    case "--ollama-url" -> options.ollamaBaseUrl = URI.create(stripTrailingSlash(args[++i]));
                    case "--working-dtype" -> options.workingDType = DType.valueOf(args[++i]);
                    case "--working-qtype" -> options.workingQType = DType.valueOf(args[++i]);
                    case "--max-tokens" -> options.maxTokens = Integer.parseInt(args[++i]);
                    case "--temperature" -> options.temperature = Float.parseFloat(args[++i]);
                    case "--seed" -> options.seed = "none".equalsIgnoreCase(args[++i]) ? null : Integer.parseInt(args[i]);
                    case "--warmup-cases" -> options.warmupCases = Integer.parseInt(args[++i]);
                    case "--max-cases" -> options.maxCases = Integer.parseInt(args[++i]);
                    case "--tensor-parallel-size" -> options.tensorParallelSize = Integer.parseInt(args[++i]);
                    case "--tensor-parallel-max-ranks-per-worker" ->
                            options.tensorParallelMaxRanksPerWorker = Integer.parseInt(args[++i]);
                    case "--suite-file" -> options.suiteFile = Path.of(args[++i]);
                    case "--output" -> options.output = Path.of(args[++i]);
                    case "--jsonl-output" -> options.jsonlOutput = Path.of(args[++i]);
                    case "--help" -> {
                        printHelpAndExit();
                    }
                    default -> throw new IllegalArgumentException("Unknown argument " + args[i]);
                }
            }
            return options;
        }

        private static String stripTrailingSlash(String value) {
            return value.endsWith("/") ? value.substring(0, value.length() - 1) : value;
        }

        private static void printHelpAndExit() {
            System.out.println("""
                    Usage: InferenceBenchmark [options]

                      --engine deliverance|ollama|both   Engine to run, default deliverance
                      --owner OWNER                      Deliverance Hugging Face owner, default tjake
                      --model MODEL                      Deliverance model, default gemma-2-2b-it-JQ4
                      --ollama-model MODEL               Ollama model name, default llama3.2
                      --ollama-url URL                   Ollama base URL, default http://localhost:11434
                      --max-tokens N                     Max generated tokens per turn, default 256
                      --temperature F                    Temperature, default 0.0
                      --seed N|none                      Seed, default 42
                      --warmup-cases N                   Cases to run before recording, default 1
                      --max-cases N                      Limit cases, default all
                      --tensor-parallel-size N           Local in-process tensor parallel ranks, default 1
                      --tensor-parallel-max-ranks-per-worker N Max ranks per local worker, default 2
                      --suite-file PATH                  FastChat MT-Bench question.jsonl; default built-in subset
                      --output PATH                      CSV output path, default target/inference-benchmark.csv
                      --jsonl-output PATH                Optional JSONL transcript output path
                    """);
            System.exit(0);
        }
    }
}
