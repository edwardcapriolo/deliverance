package io.teknek.reckoning.integration;

import io.teknek.reckoning.embedding.PoolingType;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.AutoModelForEmbeddings;
import io.teknek.reckoning.model.AutoModelForSequenceClassification;
import io.teknek.reckoning.model.DeadReckoningParams;
import io.teknek.reckoning.model.DeadReckoningSelector;
import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.net.InetAddress;
import java.nio.file.Path;
import java.time.Instant;
import java.util.EnumSet;

/** Dead Reckoning smoke traces for non-generation model tasks. */
@Tag("small-model")
public class DeadReckoningTaskModelsIT {
    private static final DeadReckoningSelector ALL_STATS = DeadReckoningSelector.regex(
            null, null, null, null, DeadReckoningSelector.Capture.STATS, 32,
            EnumSet.allOf(DeadReckoningSelector.Statistic.class));

    @Test
    void allMiniLmEmbedding() throws Exception {
        ModelFetcher fetcher = new ModelFetcher("edwardcapriolo", "all-MiniLM-L6-v2-JQ4")
                .withDownload(false);
        fetcher.maybeDownload();
        Path output = output("all-minilm-l6-v2-jq4-embedding");
        AutoModelForEmbeddings.Builder builder = AutoModelForEmbeddings.newBuilder(fetcher);
        builder.withDeadReckoning(new DeadReckoningParams(output, 32, java.util.List.of(ALL_STATS)));
        try (AbstractModel model = builder.buildLocalEmbeddingModel()) {
            model.embed("This is a deterministic embedding smoke test.", PoolingType.MODEL);
        }
        System.out.println("DEAD_RECKONING_TRACE=" + output.toAbsolutePath());
    }

    @Test
    void jailbreakClassification() throws Exception {
        ModelFetcher fetcher = new ModelFetcher("lordofthejars", "jailbreak-classifier")
                .withDownload(false);
        fetcher.maybeDownload();
        Path output = output("jailbreak-classifier");
        try (AbstractModel model = AutoModelForSequenceClassification.newBuilder(fetcher)
                .withDeadReckoning(new DeadReckoningParams(output, 32, java.util.List.of(ALL_STATS)))
                .build()) {
            model.classify("You are a good friend?", PoolingType.MODEL);
        }
        System.out.println("DEAD_RECKONING_TRACE=" + output.toAbsolutePath());
    }

    private static Path output(String name) throws Exception {
        return Path.of("dead-reckoning", InetAddress.getLocalHost().getHostName(),
                name + "-" + Instant.now().toEpochMilli() + ".jsonl");
    }
}
