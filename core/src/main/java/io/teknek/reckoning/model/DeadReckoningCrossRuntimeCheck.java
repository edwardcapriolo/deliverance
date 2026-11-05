package io.teknek.reckoning.model;

import com.fasterxml.jackson.databind.JsonNode;
import io.teknek.reckoning.JsonUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Threshold-based comparison of a dense HF trace against a Reckoning trace. */
public final class DeadReckoningCrossRuntimeCheck {
    private static final List<String> METRICS = List.of(
            "min", "max", "mean", "sum", "mean_abs", "max_abs", "l2_norm", "rms", "stddev",
            "zero_count", "nan_count", "positive_infinity_count", "negative_infinity_count");

    private DeadReckoningCrossRuntimeCheck() {
    }

    public enum Status { GREEN, YELLOW, RED }

    public record Threshold(double passAbsolute, double passRelative, double warningAbsolute,
            double warningRelative) {
        public Threshold {
            if (passAbsolute < 0 || passRelative < 0 || warningAbsolute < passAbsolute
                    || warningRelative < passRelative) {
                throw new IllegalArgumentException("Invalid cross-runtime thresholds");
            }
        }

        private Severity classify(double delta, double expected) {
            double relative = delta / Math.max(Math.abs(expected), 1.0e-12);
            if (delta <= passAbsolute || relative <= passRelative) return Severity.PASS;
            if (delta <= warningAbsolute || relative <= warningRelative) return Severity.WARNING;
            return Severity.FAIL;
        }
    }

    public record StageMapping(Pattern hfName, String reckoningName) {
        public StageMapping {
            Objects.requireNonNull(hfName, "hfName");
            Objects.requireNonNull(reckoningName, "reckoningName");
        }

        private String map(String name) {
            Matcher matcher = hfName.matcher(name);
            return matcher.matches() ? matcher.replaceFirst(reckoningName) : null;
        }
    }

    public record Policy(Threshold defaultThreshold, Map<String, Threshold> stageThresholds,
            List<StageMapping> mappings) {
        public Policy {
            Objects.requireNonNull(defaultThreshold, "defaultThreshold");
            stageThresholds = Map.copyOf(stageThresholds);
            mappings = List.copyOf(mappings);
        }

        public static Policy causalDefaults() {
            Threshold normal = new Threshold(1.0e-3, 1.0e-3, 0.25, 0.05);
            return new Policy(normal, Map.of(), List.of(
                    new StageMapping(Pattern.compile("model\\.embed_tokens"), "layer.-1.input"),
                    new StageMapping(Pattern.compile("model\\.layers\\.(\\d+)\\.input_layernorm"),
                            "layer.$1.attention_projection_input"),
                    new StageMapping(Pattern.compile("model\\.layers\\.(\\d+)\\.self_attn\\.q_proj"),
                            "layer.$1.query_projection"),
                    new StageMapping(Pattern.compile("model\\.layers\\.(\\d+)\\.self_attn\\.k_proj"),
                            "layer.$1.key_projection"),
                    new StageMapping(Pattern.compile("model\\.layers\\.(\\d+)\\.self_attn\\.v_proj"),
                            "layer.$1.value_projection"),
                    new StageMapping(Pattern.compile("model\\.layers\\.(\\d+)\\.self_attn\\.q_norm"),
                            "layer.$1.query_normalized"),
                    new StageMapping(Pattern.compile("model\\.layers\\.(\\d+)\\.self_attn\\.k_norm"),
                            "layer.$1.key_normalized"),
                    new StageMapping(Pattern.compile("model\\.layers\\.(\\d+)\\.self_attn\\.o_proj"),
                            "layer.$1.attention_output"),
                    new StageMapping(Pattern.compile("model\\.layers\\.(\\d+)\\.post_attention_layernorm"),
                            "layer.$1.post_attention_norm"),
                    new StageMapping(Pattern.compile("model\\.layers\\.(\\d+)"),
                            "layer.$1.layer_output")));
        }

        private Threshold threshold(String name) {
            return stageThresholds.getOrDefault(name, defaultThreshold);
        }
    }

    public record Result(Status status, int matchedStages, int missingStages, int unmappedStages,
            List<String> differences) {
        public boolean passed() {
            return status != Status.RED;
        }

        public String summary() {
            StringBuilder output = new StringBuilder()
                    .append("HF/Reckoning cross-runtime status=").append(status)
                    .append(" matched=").append(matchedStages)
                    .append(" missing=").append(missingStages)
                    .append(" unmapped=").append(unmappedStages).append('\n');
            differences.forEach(value -> output.append("- ").append(value).append('\n'));
            return output.toString();
        }
    }

    /** Compares traces and prints the classified report. */
    public static Result compare(Path hfTrace, Path reckoningTrace, Policy policy) {
        Map<String, JsonNode> hf = read(hfTrace);
        Map<String, JsonNode> reckoning = read(reckoningTrace);
        List<String> differences = new ArrayList<>();
        int matched = 0;
        int missing = 0;
        int unmapped = 0;
        Status status = Status.GREEN;

        for (JsonNode hfRecord : hf.values()) {
            if (!"CHECKPOINT".equals(hfRecord.path("kind").asText())) continue;
            String mappedName = mapName(hfRecord.path("name").asText(), policy.mappings());
            if (mappedName == null) {
                unmapped++;
                continue;
            }
            String key = key(hfRecord.path("phase").asText(), hfRecord.path("position").asInt(), mappedName);
            JsonNode javaRecord = reckoning.get(key);
            if (javaRecord == null) {
                missing++;
                status = Status.RED;
                differences.add("missing stage " + key);
                continue;
            }
            matched++;
            Status stageStatus = compareStage(hfRecord, javaRecord, mappedName, policy.threshold(mappedName),
                    differences);
            status = worse(status, stageStatus);
        }

        compareTokens(hf, reckoning, differences);
        if (!differences.isEmpty() && status == Status.GREEN) status = Status.YELLOW;
        Result result = new Result(status, matched, missing, unmapped, List.copyOf(differences));
        System.out.print(result.summary());
        return result;
    }

    private static Status compareStage(JsonNode hf, JsonNode java, String name, Threshold defaultThreshold,
            List<String> differences) {
        Status status = Status.GREEN;
        if (!compatibleShape(hf.path("shape"), java.path("shape"))) {
            differences.add(name + " shape HF=" + hf.path("shape") + " Java=" + java.path("shape"));
            return Status.RED;
        }
        for (String metric : METRICS) {
            if (!hf.has(metric) || !java.has(metric)) continue;
            double expected = hf.get(metric).asDouble();
            double actual = java.get(metric).asDouble();
            if (!Double.isFinite(expected) || !Double.isFinite(actual)) {
                if (Double.doubleToLongBits(expected) != Double.doubleToLongBits(actual)) {
                    differences.add(name + " " + metric + " non-finite HF=" + expected + " Java=" + actual);
                    status = Status.RED;
                }
                continue;
            }
            double delta = Math.abs(actual - expected);
            Severity severity = defaultThreshold.classify(delta, expected);
            if (severity != Severity.PASS) {
                differences.add(name + " " + metric + " delta=" + delta + " relative="
                        + delta / Math.max(Math.abs(expected), 1.0e-12) + " severity=" + severity);
                status = worse(status, severity == Severity.FAIL ? Status.RED : Status.YELLOW);
            }
        }
        return status;
    }

    private static void compareTokens(Map<String, JsonNode> hf, Map<String, JsonNode> java,
            List<String> differences) {
        JsonNode left = hf.get("RESPONSE_END");
        JsonNode right = java.get("RESPONSE_END");
        if (left != null && right != null && !left.path("generated_tokens").equals(right.path("generated_tokens"))) {
            differences.add("generated tokens differ HF=" + left.path("generated_tokens")
                    + " Java=" + right.path("generated_tokens"));
        }
    }

    private static Status worse(Status left, Status right) {
        return left.ordinal() >= right.ordinal() ? left : right;
    }

    private static String mapName(String name, List<StageMapping> mappings) {
        for (StageMapping mapping : mappings) {
            String mapped = mapping.map(name);
            if (mapped != null) return mapped;
        }
        return null;
    }

    private static boolean compatibleShape(JsonNode left, JsonNode right) {
        List<Integer> a = dimensions(left);
        List<Integer> b = dimensions(right);
        while (a.size() > 1 && a.get(0) == 1) a = a.subList(1, a.size());
        while (b.size() > 1 && b.get(0) == 1) b = b.subList(1, b.size());
        return a.equals(b);
    }

    private static List<Integer> dimensions(JsonNode node) {
        List<Integer> result = new ArrayList<>();
        node.forEach(value -> result.add(value.asInt()));
        return result;
    }

    private static String key(String phase, int position, String name) {
        return phase + "|" + position + "|" + name;
    }

    private static Map<String, JsonNode> read(Path path) {
        Map<String, JsonNode> result = new LinkedHashMap<>();
        try {
            for (String line : Files.readAllLines(path)) {
                if (line.isBlank()) continue;
                JsonNode node = JsonUtils.om.readTree(line);
                String kind = node.path("kind").asText();
                if ("CHECKPOINT".equals(kind)) {
                    result.put(key(node.path("phase").asText(), node.path("position").asInt(),
                            node.path("name").asText()), node);
                } else if ("RESPONSE_END".equals(kind)) {
                    result.put(kind, node);
                }
            }
            return result;
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read trace " + path, e);
        }
    }

    private enum Severity { PASS, WARNING, FAIL }
}
