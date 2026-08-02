package io.teknek.deliverance.tensorlib;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorLocality;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;

/** Fixed-worker executor for locality-aware tensor plan experiments. */
public final class TensorRuntime implements AutoCloseable {
    private final TensorRuntimeMode mode;
    private final TensorRuntimeNative nativeRuntime;
    private final MetricRegistry metrics;
    private final List<Worker> workers;
    private final AtomicInteger nextWorker = new AtomicInteger();
    private final LocalityStats stats = new LocalityStats();

    public TensorRuntime(int workerCount, TensorRuntimeMode mode, TensorRuntimeNative nativeRuntime,
            MetricRegistry metrics) {
        if (workerCount < 1) {
            throw new IllegalArgumentException("workerCount must be >= 1");
        }
        this.mode = Objects.requireNonNull(mode, "mode");
        this.nativeRuntime = Objects.requireNonNull(nativeRuntime, "nativeRuntime");
        this.metrics = metrics;
        mark("tensorruntime.mode." + mode.name().toLowerCase());
        mark(nativeRuntime.available() ? "tensorruntime.native.available" : "tensorruntime.native.unavailable");
        this.workers = new ArrayList<>(workerCount);
        for (int i = 0; i < workerCount; i++) {
            int cpu = nativeRuntime.cpuForWorker(i).orElse(i);
            workers.add(new Worker(i, nativeRuntime.numaNodeOfCpu(cpu).orElse(TensorLocality.UNKNOWN_NUMA_NODE), cpu));
        }
        workers.forEach(Worker::start);
    }

    public CompletableFuture<Void> submit(String operation, int chunkId, Optional<AbstractTensor> tensor, Runnable task) {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(tensor, "tensor");
        Objects.requireNonNull(task, "task");
        Optional<TensorLocality> locality = tensor.flatMap(this::localityOf);
        mark("tensorruntime.tasks.submitted");
        if (locality.map(TensorLocality::numaKnown).orElse(false)) {
            mark("tensorruntime.locality.input.known");
        } else {
            mark("tensorruntime.locality.input.unknown");
        }
        Worker worker = chooseWorker(locality);
        boolean policyApplied = policyApplied(locality, worker);
        CompletableFuture<Void> future = new CompletableFuture<>();
        worker.submit(new Work(operation, chunkId, locality, policyApplied, task, future));
        return future;
    }

    public void runAndWait(String operation, int chunkId, Optional<AbstractTensor> tensor, Runnable task) {
        submit(operation, chunkId, tensor, task).join();
    }

    public LocalitySnapshot snapshot() {
        return stats.snapshot();
    }

    public Optional<TensorLocality> ensureLocality(AbstractTensor tensor) {
        Objects.requireNonNull(tensor, "tensor");
        return localityOf(tensor);
    }

    private Worker chooseWorker(Optional<TensorLocality> locality) {
        if (mode == TensorRuntimeMode.ENFORCE && locality.isPresent() && locality.get().numaKnown()) {
            int targetNode = locality.get().numaNode();
            Optional<Worker> localWorker = workers.stream()
                    .filter(worker -> worker.numaNode == targetNode)
                    .min(Comparator.comparingInt(worker -> worker.workerId))
                    .stream().findFirst();
            if (localWorker.isPresent()) {
                mark("tensorruntime.enforce.local_worker_selected");
                return localWorker.get();
            }
            mark("tensorruntime.enforce.no_local_worker");
            return roundRobinWorker();
        }
        if (mode == TensorRuntimeMode.ENFORCE) {
            mark("tensorruntime.enforce.locality_unknown");
        }
        return roundRobinWorker();
    }

    private Worker roundRobinWorker() {
        return workers.get(Math.floorMod(nextWorker.getAndIncrement(), workers.size()));
    }

    private boolean policyApplied(Optional<TensorLocality> locality, Worker worker) {
        if (locality.map(TensorLocality::numaKnown).orElse(false)) {
            return worker.numaNode == locality.get().numaNode();
        }
        return nativeRuntime.available() && worker.affinityApplied;
    }

    private Optional<TensorLocality> localityOf(AbstractTensor tensor) {
        Optional<TensorLocality> existing = tensor.locality();
        if (existing.isPresent()) {
            return existing;
        }
        Optional<TensorLocality> observed = nativeRuntime.localityOf(tensor);
        observed.ifPresent(tensor::setLocality);
        return observed;
    }

    private void mark(String name) {
        if (metrics != null) {
            metrics.counter(name).inc();
        }
    }

    @Override
    public void close() {
        workers.forEach(Worker::stop);
    }

    private record Work(String operation, int chunkId, Optional<TensorLocality> locality, boolean policyApplied, Runnable task,
                        CompletableFuture<Void> future) {
    }

    public record LocalitySnapshot(long local, long remote, long unknown, long totalTasks, long totalRuntimeNanos) {
    }

    private final class Worker implements Runnable {
        private final int workerId;
        private final int numaNode;
        private final int cpu;
        private final LinkedBlockingQueue<Work> queue = new LinkedBlockingQueue<>();
        private final Thread thread;
        private volatile boolean running = true;
        private volatile boolean affinityApplied;

        private Worker(int workerId, int numaNode, int cpu) {
            this.workerId = workerId;
            this.numaNode = numaNode;
            this.cpu = cpu;
            this.thread = new Thread(this, "tensor-runtime-" + workerId);
            this.thread.setDaemon(true);
        }

        private void start() {
            thread.start();
        }

        private void submit(Work work) {
            queue.add(work);
        }

        private void stop() {
            running = false;
            queue.add(new Work("stop", -1, Optional.empty(), false, () -> {}, new CompletableFuture<>()));
        }

        @Override
        public void run() {
            if (nativeRuntime.available()) {
                if (nativeRuntime.pinCurrentThread(cpu)) {
                    affinityApplied = true;
                    mark("tensorruntime.affinity.pinned");
                } else {
                    mark("tensorruntime.affinity.failed");
                }
            } else {
                mark("tensorruntime.affinity.unsupported");
            }
            while (running) {
                try {
                    Work work = queue.take();
                    if (!running) {
                        return;
                    }
                    long start = System.nanoTime();
                    try {
                        work.task().run();
                    } catch (Throwable t) {
                        work.future().completeExceptionally(t);
                    } finally {
                        long elapsed = System.nanoTime() - start;
                        record(work, elapsed);
                        if (!work.future().isDone()) {
                            work.future().complete(null);
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }

        private void record(Work work, long elapsedNanos) {
            if (mode == TensorRuntimeMode.DISABLED) {
                return;
            }
            stats.totalTasks.increment();
            stats.totalRuntimeNanos.add(elapsedNanos);
            mark(work.policyApplied() ? "tensorruntime.policy.applied" : "tensorruntime.policy.not_applied");
            OptionalInt tensorNode = work.locality().map(TensorLocality::numaNode).stream().mapToInt(Integer::intValue)
                    .findFirst();
            if (tensorNode.isEmpty() || tensorNode.getAsInt() < 0 || numaNode < 0) {
                stats.unknown.increment();
                mark("tensorruntime.locality.unknown");
            } else if (tensorNode.getAsInt() == numaNode) {
                stats.local.increment();
                mark("tensorruntime.locality.local");
            } else {
                stats.remote.increment();
                mark("tensorruntime.locality.remote");
            }
        }

        private void mark(String name) {
            if (metrics != null) {
                metrics.counter(name).inc();
            }
        }
    }

    private static final class LocalityStats {
        private final LongAdder local = new LongAdder();
        private final LongAdder remote = new LongAdder();
        private final LongAdder unknown = new LongAdder();
        private final LongAdder totalTasks = new LongAdder();
        private final LongAdder totalRuntimeNanos = new LongAdder();

        private LocalitySnapshot snapshot() {
            return new LocalitySnapshot(local.sum(), remote.sum(), unknown.sum(), totalTasks.sum(),
                    totalRuntimeNanos.sum());
        }
    }
}
