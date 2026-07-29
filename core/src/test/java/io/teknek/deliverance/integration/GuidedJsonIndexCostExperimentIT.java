package io.teknek.deliverance.integration;

import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.grace.TokenIds;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.sketches.SketchesSettings;
import io.teknek.sketches.guide.Index;
import io.teknek.sketches.guide.LazyIndex;
import io.teknek.sketches.guide.Vocabulary;
import io.teknek.sketches.json.JsonSchemaRegexBuilder;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GuidedJsonIndexCostExperimentIT {

    @Test
    void estimateGuidedJsonIndexCostsWithQwen06bVocabulary() {
        String owner = System.getProperty("deliverance.guided.cost.owner", "edwardcapriolo");
        String model = System.getProperty("deliverance.guided.cost.model", "Qwen3-0.6B-JQ4");
        int maxTransitions = Integer.getInteger("deliverance.guided.cost.maxTransitions", 20_000_000);
        int maxStates = Integer.getInteger("deliverance.guided.cost.maxStates", 5_000_000);
        int maxRegexLength = Integer.getInteger("deliverance.guided.cost.maxRegexLength", 500_000);

        ModelFetcher fetcher = new ModelFetcher(owner, model).withDownload(false);
        Path modelPath = fetcher.pathForModel();
        PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(modelPath);
        Vocabulary vocabulary = vocabularyFromTokenizer(tokenizer);
        SketchesSettings settings = new SketchesSettings(maxRegexLength, maxStates, maxTransitions);

        System.out.printf("GUIDED_JSON_COST_VOCAB owner=%s model=%s vocab=%d maxRegexLength=%d maxStates=%d maxTransitions=%d%n",
                owner, model, vocabulary.size(), maxRegexLength, maxStates, maxTransitions);

        for (Sample sample : samples()) {
            long start = System.nanoTime();
            String regex;
            try {
                regex = JsonSchemaRegexBuilder.buildRegexFromSchema(sample.schema());
            } catch (RuntimeException e) {
                System.out.printf("GUIDED_JSON_COST name=%s source=%s schema=FAILED error=%s%n",
                        sample.name(), sample.source(), e.getMessage());
                continue;
            }
            long regexMs = elapsedMs(start);
            start = System.nanoTime();
            LazyIndex lazyIndex = new LazyIndex(regex, vocabulary);
            List<Integer> initialTokens = lazyIndex.getAllowedTokens(lazyIndex.getInitialState());
            assertFalse(initialTokens.isEmpty(), "lazy guide produced no initial tokens for " + sample.name());
            System.out.printf("GUIDED_JSON_COST_LAZY name=%s source=%s regexChars=%d regexMs=%d lazyInitAndFirstMaskMs=%d computedStates=%d computedTransitions=%d initialTokens=%d status=OK%n",
                    sample.name(), sample.source(), regex.length(), regexMs, elapsedMs(start),
                    lazyIndex.computedStateCount(), lazyIndex.computedTransitionCount(), initialTokens.size());

            if (!Boolean.getBoolean("deliverance.guided.cost.runEager")) {
                continue;
            }
            start = System.nanoTime();
            try {
                Index index = new Index(regex, vocabulary, settings);
                assertTrue(index.stateCount() > 0);
                System.out.printf("GUIDED_JSON_COST_EAGER name=%s source=%s regexChars=%d regexMs=%d indexMs=%d states=%d transitions=%d status=OK%n",
                        sample.name(), sample.source(), regex.length(), regexMs, elapsedMs(start), index.stateCount(),
                        index.transitionCount());
            } catch (RuntimeException e) {
                System.out.printf("GUIDED_JSON_COST_EAGER name=%s source=%s regexChars=%d regexMs=%d indexMs=%d status=FAILED error=%s%n",
                        sample.name(), sample.source(), regex.length(), regexMs, elapsedMs(start), e.getMessage());
            }
        }
    }

    private static Vocabulary vocabularyFromTokenizer(PreTrainedTokenizer tokenizer) {
        OptionalInt eos = tokenizer.eosTokenId();
        int eosTokenId = eos.orElseThrow(() -> new IllegalStateException("Tokenizer has no eos_token_id"));
        Vocabulary vocabulary = new Vocabulary(eosTokenId, Map.of());
        for (Integer tokenId : sortedTokenIds(tokenizer.getVocab())) {
            if (tokenId == eosTokenId) {
                continue;
            }
            String decoded = tokenizer.decode(new TokenIds(tokenId), false, false, false, false);
            if (!decoded.isEmpty()) {
                vocabulary.insert(decoded, tokenId);
            }
        }
        return vocabulary;
    }

    private static List<Integer> sortedTokenIds(Map<String, Integer> vocab) {
        return vocab.values().stream().distinct().sorted().toList();
    }

    private static long elapsedMs(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000L;
    }

    private static List<Sample> samples() {
        List<Sample> samples = new ArrayList<>();
        samples.add(new Sample("jsonplaceholder_todo",
                "https://jsonplaceholder.typicode.com/todos/1",
                """
                {"type":"object","additionalProperties":false,"required":["userId","id","title","completed"],"properties":{"userId":{"type":"integer"},"id":{"type":"integer"},"title":{"type":"string","maxLength":80},"completed":{"type":"boolean"}}}
                """));
        samples.add(new Sample("github_repo_summary",
                "https://api.github.com/repos/dottxt-ai/outlines",
                """
                {"type":"object","additionalProperties":false,"required":["id","node_id","name","full_name","private","owner","html_url","description","stargazers_count","license","topics"],"properties":{"id":{"type":"integer"},"node_id":{"type":"string","maxLength":64},"name":{"type":"string","maxLength":120},"full_name":{"type":"string","maxLength":180},"private":{"type":"boolean"},"owner":{"type":"object","additionalProperties":false,"required":["login","id","node_id","avatar_url","type"],"properties":{"login":{"type":"string","maxLength":120},"id":{"type":"integer"},"node_id":{"type":"string","maxLength":64},"avatar_url":{"type":"string","maxLength":240},"type":{"type":"string","maxLength":40}}},"html_url":{"type":"string","maxLength":240},"description":{"type":"string","maxLength":240},"stargazers_count":{"type":"integer"},"license":{"type":"object","additionalProperties":false,"required":["key","name","spdx_id"],"properties":{"key":{"type":"string","maxLength":40},"name":{"type":"string","maxLength":120},"spdx_id":{"type":"string","maxLength":40}}},"topics":{"type":"array","minItems":0,"maxItems":12,"items":{"type":"string","maxLength":60}}}}
                """));
        samples.add(new Sample("kubernetes_pod_shape",
                "https://kubernetes.io/docs/concepts/workloads/pods/",
                """
                {"type":"object","additionalProperties":false,"required":["apiVersion","kind","metadata","spec"],"properties":{"apiVersion":{"type":"string","maxLength":40},"kind":{"type":"string","maxLength":40},"metadata":{"type":"object","additionalProperties":false,"required":["name","labels"],"properties":{"name":{"type":"string","maxLength":80},"labels":{"type":"object","additionalProperties":false,"required":["app"],"properties":{"app":{"type":"string","maxLength":80}}}}},"spec":{"type":"object","additionalProperties":false,"required":["containers"],"properties":{"containers":{"type":"array","minItems":1,"maxItems":4,"items":{"type":"object","additionalProperties":false,"required":["name","image","ports"],"properties":{"name":{"type":"string","maxLength":80},"image":{"type":"string","maxLength":160},"ports":{"type":"array","minItems":0,"maxItems":4,"items":{"type":"object","additionalProperties":false,"required":["containerPort"],"properties":{"containerPort":{"type":"integer"}}}}}}}}}}}
                """));
        samples.add(new Sample("openapi_pet_shape",
                "https://swagger.io/specification/",
                """
                {"type":"object","additionalProperties":false,"required":["openapi","info","paths"],"properties":{"openapi":{"type":"string","maxLength":20},"info":{"type":"object","additionalProperties":false,"required":["title","version","description"],"properties":{"title":{"type":"string","maxLength":120},"version":{"type":"string","maxLength":40},"description":{"type":"string","maxLength":240}}},"paths":{"type":"object","additionalProperties":false,"required":["/pets"],"properties":{"/pets":{"type":"object","additionalProperties":false,"required":["get","post"],"properties":{"get":{"type":"object","additionalProperties":false,"required":["summary","operationId"],"properties":{"summary":{"type":"string","maxLength":120},"operationId":{"type":"string","maxLength":80}}},"post":{"type":"object","additionalProperties":false,"required":["summary","operationId"],"properties":{"summary":{"type":"string","maxLength":120},"operationId":{"type":"string","maxLength":80}}}}}}}}}
                """));
        samples.add(new Sample("blob_payload_64",
                "Synthetic base64/blob-like payload",
                blobSchema(64)));
        samples.add(new Sample("blob_payload_256",
                "Synthetic base64/blob-like payload",
                blobSchema(256)));
        samples.add(new Sample("dead_to_rights_case",
                "Deliverance Dead to Rights guided setup schema",
                """
                {"type":"object","additionalProperties":false,"required":["caseTitle","suspect","setting","meansClue","opportunityClue","mistakeClue","hiddenTruth"],"properties":{"caseTitle":{"type":"string","maxLength":80},"suspect":{"type":"string","maxLength":80},"setting":{"type":"string","maxLength":80},"meansClue":{"type":"string","maxLength":120},"opportunityClue":{"type":"string","maxLength":120},"mistakeClue":{"type":"string","maxLength":120},"hiddenTruth":{"type":"object","additionalProperties":false,"required":["crime","method","mistakes","whyCluesMatter"],"properties":{"crime":{"type":"string","maxLength":180},"method":{"type":"string","maxLength":180},"mistakes":{"type":"array","minItems":1,"maxItems":3,"items":{"type":"string","maxLength":180}},"whyCluesMatter":{"type":"array","minItems":1,"maxItems":3,"items":{"type":"string","maxLength":180}}}}}}
                """));
        return samples;
    }

    private static String blobSchema(int maxLength) {
        return """
                {"type":"object","additionalProperties":false,"required":["id","mimeType","data"],"properties":{"id":{"type":"string","maxLength":64},"mimeType":{"type":"string","maxLength":80},"data":{"type":"string","maxLength":%d}}}
                """.formatted(maxLength);
    }

    private record Sample(String name, String source, String schema) {
    }
}
