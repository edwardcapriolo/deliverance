package io.teknek.deliverance.math;

import com.google.common.base.Suppliers;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

/**
 * Executor that sizes based on cpu
 */
public class PhysicalCoreTuningExecutor {

    // I can get behind this ONE singleton since it is ideally tied to number of cores
    // but it will go away soon as it will fall apart when trying to run two instances same vm

    public static AtomicReference<Integer> overrideCores = new AtomicReference<>(null);
    public static final Supplier<PhysicalCoreTuningExecutor> instance = Suppliers.memoize(
            () -> overrideCores.get() == null ? new PhysicalCoreTuningExecutor() :
                    new PhysicalCoreTuningExecutor(overrideCores.get()));

    private final ForkJoinPool pool;

    public PhysicalCoreTuningExecutor(int cores) {
        int available = Runtime.getRuntime().availableProcessors();
        if (cores < 1){
            throw new IllegalArgumentException("cores must be > 0 ");
        }
        if (cores > Runtime.getRuntime().availableProcessors()) {
            throw new IllegalArgumentException("cores must be less than or equal to " + available);
        }
        this.pool = new ForkJoinPool(cores, ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true);
    }

    public PhysicalCoreTuningExecutor(){
        int cores = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
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
