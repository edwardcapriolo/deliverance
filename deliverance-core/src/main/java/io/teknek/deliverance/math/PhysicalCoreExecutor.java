package io.teknek.deliverance.math;


import com.google.common.base.Suppliers;
import net.jafama.FastMath;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;

/**
 * Executor that uses a fixed number of physical cores
 */
public class PhysicalCoreExecutor {
    private static volatile int physicalCoreCount = Math.max(2, Runtime.getRuntime().availableProcessors() / 2);
    private static final AtomicBoolean started = new AtomicBoolean(false);

    /**
     * Override the number of physical cores to use
     * @param threadCount number of physical cores to use
     */
    public static void overrideThreadCount(int threadCount) {
        assert threadCount > 0 && threadCount <= Runtime.getRuntime().availableProcessors() : "Threads must be < cores: " + threadCount;

        if (!started.compareAndSet(false, true)) throw new IllegalStateException("Executor already started");

        physicalCoreCount = threadCount;
    }

    public static final Supplier<PhysicalCoreExecutor> instance = Suppliers.memoize(() -> {
        started.set(true);
        return new PhysicalCoreExecutor(physicalCoreCount);
    });

    private final ForkJoinPool pool;

    private PhysicalCoreExecutor(int cores) {
        assert cores > 0 && cores <= Runtime.getRuntime().availableProcessors() : "Invalid core count: " + cores;
        this.pool = new ForkJoinPool(cores, ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true);
    }

    public void execute(Runnable run) {
        pool.submit(run).join();
    }

    public <T> T submit(Supplier<T> run) {
        return pool.submit(run::get).join();
    }

    public int getCoreCount() {
        return pool.getParallelism();
    }
}
