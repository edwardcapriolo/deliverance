package io.teknek.deliverance.model;

import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
}
