package io.teknek.deliverance.model;

import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.SingleRankTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.GossipParallelSettings;
import io.teknek.deliverance.model.tensorparallel.TensorParallelDeploymentSpec;
import io.teknek.gossip.GossipSettings;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AutoModelForCausaLmTensorParallelTest {

    @Test
    public void builderDefaultsToSingleRankTensorParallelContext() {
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(new ModelFetcher("owner", "model"));

        assertEquals(0, builder.getTensorParallelContext().rank());
        assertEquals(1, builder.getTensorParallelContext().size());
        assertFalse(builder.getTensorParallelContext().enabled());
        assertTrue(builder.getTensorParallelContext().coordinatorRank());
    }

    @Test
    public void builderAcceptsExplicitTensorParallelContext() {
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(new ModelFetcher("owner", "model"))
                .withTensorParallelContext(new StaticTensorParallelContext(2, 4));

        assertEquals(2, builder.getTensorParallelContext().rank());
        assertEquals(4, builder.getTensorParallelContext().size());
        assertTrue(builder.getTensorParallelContext().enabled());
        assertFalse(builder.getTensorParallelContext().coordinatorRank());
    }

    @Test
    public void builderConvenienceMethodCreatesStaticContext() {
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(new ModelFetcher("owner", "model"))
                .withTensorParallel(1, 2);

        assertEquals(1, builder.getTensorParallelContext().rank());
        assertEquals(2, builder.getTensorParallelContext().size());
    }

    @Test
    public void builderRejectsNullContext() {
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(new ModelFetcher("owner", "model"));

        assertThrows(NullPointerException.class, () -> builder.withTensorParallelContext(null));
    }

    @Test
    public void builderAcceptsExplicitCollectives() {
        SingleRankTensorParallelCollectives collectives = new SingleRankTensorParallelCollectives();

        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(new ModelFetcher("owner", "model"))
                .withTensorParallelCollectives(collectives);

        assertSame(collectives, builder.getTensorParallelCollectives());
    }

    @Test
    public void builderRejectsNullCollectives() {
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(new ModelFetcher("owner", "model"));

        assertThrows(NullPointerException.class, () -> builder.withTensorParallelCollectives(null));
    }

    @Test
    public void builderRejectsStartingMembershipWithoutParallelSettings() {
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(new ModelFetcher("owner", "model"));

        assertThrows(IllegalStateException.class, builder::startParallelMembership);
    }

    @Test
    public void builderStoresParallelSettings() throws Exception {
        GossipParallelSettings settings = new GossipParallelSettings("cluster", "node-0",
                new URI("udp://127.0.0.1:41000"), List.of(), new GossipSettings(),
                new TensorParallelDeploymentSpec("demo", "gemma2", 2, 1));

        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(new ModelFetcher("owner", "model"))
                .withParallelSettings(settings);

        assertEquals(settings, builder.getParallelSettings().orElseThrow());
    }
}
