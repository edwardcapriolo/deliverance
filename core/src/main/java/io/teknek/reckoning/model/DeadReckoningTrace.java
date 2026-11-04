package io.teknek.reckoning.model;

import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.teknek.reckoning.JsonUtils;
import io.teknek.reckoning.generator.GeneratorParameters;
import io.teknek.reckoning.generator.Response;
import io.teknek.reckoning.safetensors.prompt.PromptContext;
import io.teknek.reckoning.tensor2.TensorRef;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Map;
import java.util.SortedMap;

final class DeadReckoningTrace implements AutoCloseable {
    private final BufferedWriter writer;
    private final int maxValues;
    private final java.util.List<DeadReckoningSelector> selectors;
    private String phase = "PREFILL";
    private int position = -1;

    DeadReckoningTrace(DeadReckoningParams params) {
        try {
            Path output = params.output().toAbsolutePath();
            if (output.getParent() != null) {
                Files.createDirectories(output.getParent());
            }
            writer = Files.newBufferedWriter(output);
        } catch (IOException e) {
            throw new RuntimeException("Unable to open Dead Reckoning trace " + params.output(), e);
        }
        maxValues = params.traceMaxValues();
        selectors = params.selectors();
    }

    synchronized void recordLayer(AbstractModel.LayerDebugEvent event) {
        try {
            TensorRef tensor = event.hiddenStates();
            ObjectNode record = JsonUtils.om.createObjectNode();
            record.put("kind", "CHECKPOINT");
            record.put("phase", phase);
            record.put("position", position);
            record.put("layer", event.layerIndex());
            record.put("stage", event.stage());
            record.put("dtype", tensor.dType().name());
            String name = "layer." + event.layerIndex() + "." + event.stage();
            DeadReckoningSelector selector = selectors.stream()
                    .filter(value -> value.matches("CHECKPOINT", name, phase, position))
                    .findFirst().orElse(null);
            if (!selectors.isEmpty() && selector == null) return;
            record.put("name", name);
            ArrayNode shape = record.putArray("shape");
            for (int dimension : tensor.shape().shapeArray()) {
                shape.add(dimension);
            }
            DeadReckoningSelector.Capture capture = selector == null
                    ? DeadReckoningSelector.Capture.FIRST_VALUES : selector.capture();
            if (capture == DeadReckoningSelector.Capture.FIRST_VALUES || capture == DeadReckoningSelector.Capture.FULL) {
                ArrayNode values = record.putArray("values");
                int requested = selector == null ? maxValues : selector.maxValues();
                int count = capture == DeadReckoningSelector.Capture.FULL || requested == 0
                        ? Math.toIntExact(tensor.shape().size()) : Math.min(requested, Math.toIntExact(tensor.shape().size()));
                int columns = tensor.shape().last();
                for (int index = 0; index < count; index++) {
                    values.add(tensor.get(index / columns, index % columns));
                }
            } else if (capture == DeadReckoningSelector.Capture.STATS) {
                addStats(record, tensor, selector == null
                        ? java.util.Set.of(DeadReckoningSelector.Statistic.values()) : selector.statistics());
            } else {
                record.put("hash", hash(tensor));
            }
            writer.write(JsonUtils.om.writeValueAsString(record));
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            throw new RuntimeException("Unable to write Dead Reckoning trace", e);
        }
    }

    private static void addStats(ObjectNode record, TensorRef tensor,
            java.util.Set<DeadReckoningSelector.Statistic> requested) {
        double sum = 0.0;
        double sumSquares = 0.0;
        double sumAbs = 0.0;
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        float maxAbs = 0.0f;
        int nonFinite = 0;
        int nan = 0;
        int positiveInfinity = 0;
        int negativeInfinity = 0;
        int zeros = 0;
        int columns = tensor.shape().last();
        for (int index = 0; index < tensor.shape().size(); index++) {
            float value = tensor.get(index / columns, index % columns);
            if (!Float.isFinite(value)) {
                nonFinite++;
                if (Float.isNaN(value)) nan++;
                else if (value > 0) positiveInfinity++;
                else negativeInfinity++;
                continue;
            }
            if (value == 0.0f) zeros++;
            min = Math.min(min, value);
            max = Math.max(max, value);
            sum += value;
            sumAbs += Math.abs(value);
            maxAbs = Math.max(maxAbs, Math.abs(value));
            sumSquares += (double) value * value;
        }
        double count = tensor.shape().size() - nonFinite;
        double mean = count == 0 ? Double.NaN : sum / count;
        double rms = count == 0 ? Double.NaN : Math.sqrt(sumSquares / count);
        double variance = count == 0 ? Double.NaN : Math.max(0.0, sumSquares / count - mean * mean);
        put(requested, record, DeadReckoningSelector.Statistic.COUNT, "count", tensor.shape().size());
        put(requested, record, DeadReckoningSelector.Statistic.MIN, "min", count == 0 ? Double.NaN : min);
        put(requested, record, DeadReckoningSelector.Statistic.MAX, "max", count == 0 ? Double.NaN : max);
        put(requested, record, DeadReckoningSelector.Statistic.MEAN, "mean", mean);
        put(requested, record, DeadReckoningSelector.Statistic.SUM, "sum", sum);
        put(requested, record, DeadReckoningSelector.Statistic.MEAN_ABS, "mean_abs", count == 0 ? Double.NaN : sumAbs / count);
        put(requested, record, DeadReckoningSelector.Statistic.MAX_ABS, "max_abs", maxAbs);
        put(requested, record, DeadReckoningSelector.Statistic.L2_NORM, "l2_norm", Math.sqrt(sumSquares));
        put(requested, record, DeadReckoningSelector.Statistic.RMS, "rms", rms);
        put(requested, record, DeadReckoningSelector.Statistic.STDDEV, "stddev", Math.sqrt(variance));
        put(requested, record, DeadReckoningSelector.Statistic.ZERO_COUNT, "zero_count", zeros);
        put(requested, record, DeadReckoningSelector.Statistic.NAN_COUNT, "nan_count", nan);
        put(requested, record, DeadReckoningSelector.Statistic.POSITIVE_INFINITY_COUNT,
                "positive_infinity_count", positiveInfinity);
        put(requested, record, DeadReckoningSelector.Statistic.NEGATIVE_INFINITY_COUNT,
                "negative_infinity_count", negativeInfinity);
        if (requested.contains(DeadReckoningSelector.Statistic.HASH)) {
            record.put("hash", hash(tensor));
        }
        if (requested.contains(DeadReckoningSelector.Statistic.FIRST_VALUES)) {
            ArrayNode values = record.putArray("first_values");
            int sampleColumns = tensor.shape().last();
            int sampleCount = Math.min(32, Math.toIntExact(tensor.shape().size()));
            for (int index = 0; index < sampleCount; index++) {
                values.add(tensor.get(index / sampleColumns, index % sampleColumns));
            }
        }
    }

    private static void put(java.util.Set<DeadReckoningSelector.Statistic> requested, ObjectNode record,
            DeadReckoningSelector.Statistic statistic, String name, Number value) {
        if (requested.contains(statistic)) record.put(name, value.doubleValue());
    }

    private static String hash(TensorRef tensor) {
        final java.security.MessageDigest digest;
        try {
            digest = java.security.MessageDigest.getInstance("SHA-256");
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable", e);
        }
        int columns = tensor.shape().last();
        for (int index = 0; index < tensor.shape().size(); index++) {
            int bits = Float.floatToRawIntBits(tensor.get(index / columns, index % columns));
            digest.update((byte) bits);
            digest.update((byte) (bits >>> 8));
            digest.update((byte) (bits >>> 16));
            digest.update((byte) (bits >>> 24));
        }
        return java.util.HexFormat.of().formatHex(digest.digest());
    }

    private static void addStats(ObjectNode record, float[] values) {
        double sum = 0.0;
        double sumSquares = 0.0;
        double sumAbs = 0.0;
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        float maxAbs = 0.0f;
        int nan = 0;
        int positiveInfinity = 0;
        int negativeInfinity = 0;
        int zeros = 0;
        for (float value : values) {
            if (!Float.isFinite(value)) {
                if (Float.isNaN(value)) nan++;
                else if (value > 0) positiveInfinity++;
                else negativeInfinity++;
                continue;
            }
            if (value == 0.0f) zeros++;
            min = Math.min(min, value);
            max = Math.max(max, value);
            sum += value;
            sumAbs += Math.abs(value);
            maxAbs = Math.max(maxAbs, Math.abs(value));
            sumSquares += (double) value * value;
        }
        int nonFinite = nan + positiveInfinity + negativeInfinity;
        double count = values.length - nonFinite;
        double mean = count == 0 ? Double.NaN : sum / count;
        double rms = count == 0 ? Double.NaN : Math.sqrt(sumSquares / count);
        double variance = count == 0 ? Double.NaN : Math.max(0.0, sumSquares / count - mean * mean);
        record.put("count", values.length);
        record.put("min", count == 0 ? Double.NaN : min);
        record.put("max", count == 0 ? Double.NaN : max);
        record.put("mean", mean);
        record.put("sum", sum);
        record.put("mean_abs", count == 0 ? Double.NaN : sumAbs / count);
        record.put("max_abs", maxAbs);
        record.put("l2_norm", Math.sqrt(sumSquares));
        record.put("rms", rms);
        record.put("stddev", Math.sqrt(variance));
        record.put("zero_count", zeros);
        record.put("nan_count", nan);
        record.put("positive_infinity_count", positiveInfinity);
        record.put("negative_infinity_count", negativeInfinity);
        record.put("hash", hash(values));
        ArrayNode firstValues = record.putArray("first_values");
        for (int i = 0; i < Math.min(32, values.length); i++) firstValues.add(values[i]);
    }

    private static String hash(float[] values) {
        final java.security.MessageDigest digest;
        try {
            digest = java.security.MessageDigest.getInstance("SHA-256");
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable", e);
        }
        for (float value : values) {
            int bits = Float.floatToRawIntBits(value);
            digest.update((byte) bits);
            digest.update((byte) (bits >>> 8));
            digest.update((byte) (bits >>> 16));
            digest.update((byte) (bits >>> 24));
        }
        return java.util.HexFormat.of().formatHex(digest.digest());
    }

    synchronized void recordHeader(String owner, String model, PromptContext prompt,
            GeneratorParameters parameters) {
        ObjectNode record = JsonUtils.om.createObjectNode();
        record.put("kind", "RUN_START");
        record.put("owner", owner);
        record.put("model", model);
        record.put("prompt", prompt.getPrompt());
        ObjectNode generator = record.putObject("generator_parameters");
        generator.put("temperature", parameters.temperature.orElse(null));
        generator.put("max_tokens", parameters.maxTokens.orElse(null));
        generator.put("seed", parameters.seed.orElse(null));
        generator.put("log_probs", parameters.logProbs.orElse(null));
        generator.put("top_log_probs", parameters.topLogProbs.orElse(null));
        write(record);
        ObjectNode prefill = JsonUtils.om.createObjectNode();
        prefill.put("kind", "PREFILL_START");
        prefill.put("prompt_tokens", -1);
        write(prefill);
    }

    synchronized void recordTaskStart(String task, String input, String poolingType) {
        ObjectNode record = JsonUtils.om.createObjectNode();
        record.put("kind", "TASK_START");
        record.put("task", task);
        record.put("input", input);
        record.put("pooling_type", poolingType);
        write(record);
    }

    synchronized void recordEmbeddingResult(float[] embedding) {
        ObjectNode record = JsonUtils.om.createObjectNode();
        record.put("kind", "EMBEDDING_RESULT");
        record.putArray("shape").add(embedding.length);
        record.put("dtype", "F32");
        addStats(record, embedding);
        write(record);
    }

    synchronized void recordClassificationResult(SortedMap<String, Float> scores) {
        ObjectNode record = JsonUtils.om.createObjectNode();
        record.put("kind", "CLASSIFICATION_RESULT");
        ObjectNode values = record.putObject("scores");
        String selectedLabel = null;
        float selectedScore = Float.NEGATIVE_INFINITY;
        for (Map.Entry<String, Float> entry : scores.entrySet()) {
            values.put(entry.getKey(), entry.getValue());
            if (entry.getValue() > selectedScore) {
                selectedLabel = entry.getKey();
                selectedScore = entry.getValue();
            }
        }
        record.put("selected_label", selectedLabel);
        record.put("selected_score", selectedScore);
        write(record);
    }

    synchronized void recordGenerationDebug(AbstractModel.GenerationDebugEvent event) {
        if (event.type() != AbstractModel.GenerationDebugEventType.AFTER_PROMPT_PREFILL) {
            return;
        }
        ObjectNode end = JsonUtils.om.createObjectNode();
        end.put("kind", "PREFILL_END");
        end.put("prompt_tokens", event.promptTokens().length);
        end.put("last_position", event.promptTokens().length - 1);
        write(end);
        phase = "DECODE";
        position = event.promptTokens().length;
        ObjectNode start = JsonUtils.om.createObjectNode();
        start.put("kind", "DECODE_START");
        start.put("position", position);
        write(start);
    }

    synchronized void recordResponse(Response response) {
        ObjectNode record = JsonUtils.om.createObjectNode();
        record.put("kind", "response");
        record.put("text", response.responseText);
        record.put("text_with_special_tokens", response.responseTextWithSpecialTokens);
        record.put("prompt_tokens", response.promptTokens);
        record.put("finish_reason", response.finishReason.name());
        ArrayNode tokens = record.putArray("generated_tokens");
        response.generatedTokens.forEach(tokens::add);
        ArrayNode samplers = record.putArray("sampler_returns");
        response.samplerReturns.forEach(result -> {
            ObjectNode selected = JsonUtils.om.createObjectNode();
            selected.put("kind", "TOKEN_SELECTED");
            selected.put("position", position++);
            selected.put("token", result.getToken());
            write(selected);
            ObjectNode sampler = samplers.addObject();
            sampler.put("token", result.getToken());
            ArrayNode top = sampler.putArray("top_log_probs");
            result.getTopNLogProbs().ifPresent(values -> values.stream()
                    .sorted(Comparator.comparingDouble((IndexValueToken value) -> value.value).reversed())
                    .forEach(value -> {
                ObjectNode entry = top.addObject();
                entry.put("index", value.index);
                entry.put("token", value.token);
                entry.put("logit", value.value);
                entry.put("log_prob", value.logProb);
            }));
        });
        ObjectNode decodeEnd = JsonUtils.om.createObjectNode();
        decodeEnd.put("kind", "DECODE_END");
        decodeEnd.put("generated_tokens", response.generatedTokens.size());
        write(decodeEnd);
        record.put("kind", "RESPONSE_END");
        write(record);
    }

    private void write(ObjectNode record) {
        try {
            writer.write(JsonUtils.om.writeValueAsString(record));
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            throw new RuntimeException("Unable to write Dead Reckoning trace", e);
        }
    }

    @Override
    public synchronized void close() {
        try {
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException("Unable to close Dead Reckoning trace", e);
        }
    }
}
