package io.teknek.deliverance.math;

import java.util.concurrent.ForkJoinPool;
import java.util.function.Supplier;

public class WrappedForkJoinPool implements AutoCloseable{
    private final ForkJoinPool underlying;
    public WrappedForkJoinPool(ForkJoinPool underlying){
        this.underlying = underlying;
    }
    public void executeBlocking(Runnable run) {
        underlying.submit(run).join();
    }

    public <T> T submitBlocking(Supplier<T> run) {
        return underlying.submit(run::get).join();
    }

    public ForkJoinPool getUnderlying(){
        return underlying;
    }

    public int getCoreCount(){
        return underlying.getParallelism();
    }

    @Override
    public void close() {
        underlying.close();
    }

    public static ForkJoinPool autoSizeByCores(){
        int cores = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
        return new ForkJoinPool(cores, ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true);
    }
}
