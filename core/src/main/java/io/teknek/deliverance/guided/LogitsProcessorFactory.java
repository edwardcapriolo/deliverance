package io.teknek.deliverance.guided;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.common.primitives.Ints;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.sketches.SketchesSettings;
import io.teknek.sketches.guide.ChoiceGuide;
import io.teknek.sketches.guide.Guide;
import io.teknek.sketches.guide.Index;
import io.teknek.sketches.guide.IndexGuide;
import io.teknek.sketches.guide.Vocabulary;
import io.teknek.sketches.json.JsonSchemaRegexBuilder;
import io.teknek.sketches.types.Choice;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.WeakHashMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

public class LogitsProcessorFactory {
    private static final Map<AbstractModel, ConcurrentMap<IndexCacheKey, Index>> INDEX_CACHE =
            Collections.synchronizedMap(new WeakHashMap<>());
    private static final ConcurrentMap<JsonRegexCacheKey, String> JSON_REGEX_CACHE = new ConcurrentHashMap<>();

    private LogitsProcessorFactory() {
    }

    public static Optional<LogitsProcessor> create(AbstractModel model, GeneratorParameters parameters) {
        return create(model, parameters, SketchesSettings.DEFAULT);
    }

    public static Optional<LogitsProcessor> create(AbstractModel model, GeneratorParameters parameters,
            SketchesSettings settings) {
        MetricRegistry metrics = model.getMetricRegistry();
        try (Timer.Context ignored = InferenceProfiler.timer(metrics, "guided.factory").time()) {
        int guidanceModes = 0;
        guidanceModes += parameters.guidedChoice.isPresent() ? 1 : 0;
        guidanceModes += parameters.guidedRegex.isPresent() ? 1 : 0;
        guidanceModes += parameters.guidedJson.isPresent() ? 1 : 0;
        if (guidanceModes > 1) {
            throw new IllegalArgumentException("Only one guided mode can be set");
        }
        if (parameters.guidedChoice.isPresent()) {
            metrics.meter("guided.mode.choice").mark();
            Choice choice = new Choice(parameters.guidedChoice.get());
            Guide guide = new ChoiceGuide(encodeChoices(model, choice), model.getConfig().eosTokens);
            return Optional.of(new GuideLogitsProcessor(guide, metrics));
        }
        if (parameters.guidedRegex.isPresent()) {
            metrics.meter("guided.mode.regex").mark();
            Index index = indexFor(model, parameters.guidedRegex.get(), settings);
            return Optional.of(new GuideLogitsProcessor(new IndexGuide(index), metrics));
        }
        if (parameters.guidedJson.isPresent()) {
            metrics.meter("guided.mode.json").mark();
            String regex = regexForJsonSchema(model, parameters.guidedJson.get());
            Index index = indexFor(model, regex, settings);
            return Optional.of(new GuideLogitsProcessor(new IndexGuide(index), metrics));
        }
        return Optional.empty();
        }
    }

    private static Map<String, List<Integer>> encodeChoices(AbstractModel model, Choice choice) {
        Map<String, List<Integer>> encoded = new LinkedHashMap<>();
        for (String item : choice.items()) {
            List<Integer> ids = Arrays.stream(model.encodeForRuntime(item))
                    .mapToInt(Ints::checkedCast)
                    .boxed()
                    .toList();
            encoded.put(item, ids);
        }
        return encoded;
    }

    private static Vocabulary vocabularyFromModel(AbstractModel model) {
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), "guided.vocabulary_build").time()) {
            Vocabulary vocabulary = new Vocabulary(model.getConfig().eosTokens, Map.of());
            for (int tokenId = 0; tokenId < model.getConfig().vocabularySize; tokenId++) {
                if (model.getConfig().eosTokens.contains(tokenId)) {
                    continue;
                }
                vocabulary.insert(model.decodeToken(tokenId), tokenId);
            }
            model.getMetricRegistry().histogram("guided.vocabulary.size").update(vocabulary.size());
            return vocabulary;
        }
    }

    private static String regexForJsonSchema(AbstractModel model, String jsonSchema) {
        JsonRegexCacheKey key = new JsonRegexCacheKey(jsonSchema);
        String cached = JSON_REGEX_CACHE.get(key);
        if (cached != null) {
            model.getMetricRegistry().meter("guided.json_regex_cache.hit").mark();
            InferenceProfiler.counter(model.getMetricRegistry(), "guided.json_regex_cache.hit.count").inc();
            return cached;
        }
        model.getMetricRegistry().meter("guided.json_regex_cache.miss").mark();
        InferenceProfiler.counter(model.getMetricRegistry(), "guided.json_regex_cache.miss.count").inc();
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), "guided.json_schema_to_regex").time()) {
            String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(jsonSchema);
            model.getMetricRegistry().histogram("guided.regex.length").update(regex.length());
            JSON_REGEX_CACHE.putIfAbsent(key, regex);
            return regex;
        }
    }

    private static Index indexFor(AbstractModel model, String regex, SketchesSettings settings) {
        model.getMetricRegistry().histogram("guided.regex.length").update(regex.length());
        IndexCacheKey key = new IndexCacheKey(regex, settings);
        ConcurrentMap<IndexCacheKey, Index> modelCache;
        synchronized (INDEX_CACHE) {
            modelCache = INDEX_CACHE.computeIfAbsent(model, ignored -> new ConcurrentHashMap<>());
        }
        Index cached = modelCache.get(key);
        if (cached != null) {
            model.getMetricRegistry().meter("guided.index_cache.hit").mark();
            InferenceProfiler.counter(model.getMetricRegistry(), "guided.index_cache.hit.count").inc();
            recordIndexShape(model.getMetricRegistry(), cached);
            return cached;
        }
        model.getMetricRegistry().meter("guided.index_cache.miss").mark();
        InferenceProfiler.counter(model.getMetricRegistry(), "guided.index_cache.miss.count").inc();
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), "guided.index_build").time()) {
            Vocabulary vocabulary = vocabularyFromModel(model);
            Index index = new Index(regex, vocabulary, settings);
            recordIndexShape(model.getMetricRegistry(), index);
            Index existing = modelCache.putIfAbsent(key, index);
            return existing == null ? index : existing;
        }
    }

    private static void recordIndexShape(MetricRegistry metrics, Index index) {
        metrics.histogram("guided.index.states").update(index.stateCount());
        metrics.histogram("guided.index.transitions").update(index.transitionCount());
        if (InferenceProfiler.isEnabled()) {
            InferenceProfiler.counter(metrics, "guided.index.states.total").inc(index.stateCount());
            InferenceProfiler.counter(metrics, "guided.index.transitions.total").inc(index.transitionCount());
        }
    }

    private record IndexCacheKey(String regex, SketchesSettings settings) {
    }

    private record JsonRegexCacheKey(String schema) {
    }
}
