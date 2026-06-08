package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;

import java.time.Duration;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class InProcessTensorParallelCollectives implements TensorParallelCollectives {
    private final TensorParallelContext context;
    private final Group group;

    public InProcessTensorParallelCollectives(TensorParallelContext context, Group group) {
        this.context = context;
        this.group = group;
    }

    @Override
    public AbstractTensor allReduceSum(String key, AbstractTensor local) {
        return group.allReduceSum(key, context.rank(), context.size(), local);
    }

    public static class Group {
        private final Duration timeout;
        private final Map<String, Round> rounds = new HashMap<>();

        public Group(Duration timeout) {
            this.timeout = timeout;
        }

        public synchronized AbstractTensor allReduceSum(String key, int rank, int size, AbstractTensor local) {
            if (size < 2) {
                throw new IllegalArgumentException("in-process collectives require at least 2 ranks");
            }
            if (rank < 0 || rank >= size) {
                throw new IllegalArgumentException("rank must be between 0 and size - 1");
            }
            Round round = rounds.computeIfAbsent(key, ignored -> new Round(size, local));
            round.add(rank, local);
            if (round.arrived == size) {
                round.reduce();
                notifyAll();
            }
            waitForReduction(key, round);
            AbstractTensor result = round.copyReduced();
            round.returned++;
            if (round.returned == size) {
                round.closeReduced();
                rounds.remove(key);
            }
            return result;
        }

        private void waitForReduction(String key, Round round) {
            long deadline = System.nanoTime() + timeout.toNanos();
            while (round.reduced == null) {
                long remainingNanos = deadline - System.nanoTime();
                if (remainingNanos <= 0) {
                    throw new IllegalStateException("Timed out waiting for allReduceSum key=" + key
                            + " expected=" + round.size + " arrived=" + round.arrived);
                }
                try {
                    long millis = Math.max(1L, remainingNanos / 1_000_000L);
                    wait(millis);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("Interrupted waiting for allReduceSum key=" + key, e);
                }
            }
        }
    }

    private static class Round {
        private final int size;
        private final int[] shape;
        private final DType dType;
        private final AbstractTensor[] contributions;
        private int arrived;
        private int returned;
        private AbstractTensor reduced;

        private Round(int size, AbstractTensor first) {
            this.size = size;
            this.shape = first.shape().shapeArray();
            this.dType = first.dType();
            this.contributions = new AbstractTensor[size];
        }

        private void add(int rank, AbstractTensor local) {
            if (contributions[rank] != null) {
                throw new IllegalStateException("rank " + rank + " already contributed to this allReduceSum");
            }
            if (local.dType() != dType || !Arrays.equals(local.shape().shapeArray(), shape)) {
                throw new IllegalArgumentException("allReduceSum contributions must have matching dtype and shape");
            }
            contributions[rank] = copy(local);
            arrived++;
        }

        private void reduce() {
            reduced = new FloatBufferTensor(shape);
            for (AbstractTensor contribution : contributions) {
                for (int row = 0; row < reduced.shape().first(); row++) {
                    for (int col = 0; col < reduced.shape().last(); col++) {
                        reduced.set(reduced.get(row, col) + contribution.get(row, col), row, col);
                    }
                }
                contribution.close();
            }
        }

        private AbstractTensor copyReduced() {
            return copy(reduced);
        }

        private void closeReduced() {
            if (reduced != null) {
                reduced.close();
                reduced = null;
            }
        }

        private static AbstractTensor copy(AbstractTensor source) {
            if (source.dType() != DType.F32) {
                throw new UnsupportedOperationException("InProcessTensorParallelCollectives currently supports F32 tensors");
            }
            AbstractTensor copy = new FloatBufferTensor(source.shape());
            copy.copyFrom(source, 0, 0, (int) source.size());
            return copy;
        }
    }
}
