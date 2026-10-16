package io.teknek.deliverance.math;

import org.junit.jupiter.api.Test;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the {@code WrappedForkJoinPool} class, which provides a wrapper around
 * {@link ForkJoinPool} to facilitate blocking task execution and result submission. These tests
 * cover various scenarios, including executing runnables and returning supplier results,
 * reporting underlying parallelism, and closing the pool.
 */
class WrappedForkJoinPoolTest {

    @Test
    void executesRunnableAndReturnsSupplierResult() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(2))) {
            AtomicInteger value = new AtomicInteger();

            pool.executeBlocking(() -> value.set(41));

            assertEquals(41, value.get());
            assertEquals("answer", pool.submitBlocking(() -> "answer"));
        }
    }

    @Test
    void reportsUnderlyingParallelismAndClosesIt() {
        ForkJoinPool underlying = new ForkJoinPool(3);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(underlying);

        assertEquals(3, pool.getCoreCount());
        assertEquals(underlying, pool.getUnderlying());

        pool.close();

        assertTrue(underlying.isShutdown());
    }

    @Test
    void propagatesFailuresFromBlockingOperations() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(1))) {
            IllegalArgumentException runnableFailure = new IllegalArgumentException("runnable failed");
            IllegalStateException supplierFailure = new IllegalStateException("supplier failed");

            IllegalArgumentException thrownRunnable = assertThrows(IllegalArgumentException.class,
                    () -> pool.executeBlocking(() -> { throw runnableFailure; }));
            IllegalStateException thrownSupplier = assertThrows(IllegalStateException.class,
                    () -> pool.submitBlocking(() -> { throw supplierFailure; }));

            assertEquals(runnableFailure, thrownRunnable.getCause());
            assertEquals(supplierFailure, thrownSupplier.getCause());
        }
    }

    @Test
    void autoSizesToAtLeastOneWorker() {
        ForkJoinPool pool = WrappedForkJoinPool.autoSizeByCores();

        try (pool) {
            assertTrue(pool.getParallelism() >= 1);
        }
    }
}
