package io.teknek.deliverance.math;

import java.util.function.IntConsumer;
import java.util.stream.IntStream;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VectorMath {

    private static final Logger logger = LoggerFactory.getLogger(VectorMath.class);

    public static void pfor(int start, int end, IntConsumer action, WrappedForkJoinPool pool) {
        pool.executeBlocking(() -> IntStream.range(start, end).parallel().forEach(action));
    }

    /**
     * Splits a range into parts of (start, end) then hands each part to a BiIntConsumer (action)
     * @param offset a starting offset
     * @param length a length of items to split starting from the offset
     * @param action an action to perform on each split
     * @param splitSize split the list into this many parts
     */
    public static void pchunk(int offset, int length, BiIntConsumer action, int splitSize, WrappedForkJoinPool pool) {
        int splits = Math.min(length, splitSize);
        int chunkSize = length / splits;

        if (splits == 1) {
            action.accept(offset, length);
        } else {
            int remainder = length % chunkSize;

            int fsplits = splits;
            int fchunkSize = chunkSize;
            int fremainder = remainder;

            pool.executeBlocking(
                            () -> IntStream.range(0, fsplits)
                                    .parallel()
                                    .forEach(
                                            i -> action.accept(
                                                    offset + (i * fchunkSize),
                                                    fremainder > 0 && i == fsplits - 1 ? fchunkSize + fremainder : fchunkSize
                                            )
                                    )
                    );

        }
    }


    /**
     * Splits a range into parts of (start, end) then hands each part to a BiIntConsumer (action)
     * @param offset a starting offset
     * @param length a length of items to split starting from the offset
     * @param action an action to perform on each split
     * @param splitSize split the list into this many parts
     */
    public static void pchunkMetrics(int offset, int length, BiIntConsumer action, int splitSize,
                                     Timer timer, WrappedForkJoinPool pool) {
        int splits = Math.min(length, splitSize);
        int chunkSize = length / splits;

        if (splits == 1) {
            action.accept(offset, length);
        } else {
            int remainder = length % chunkSize;

            int fsplits = splits;
            int fchunkSize = chunkSize;
            int fremainder = remainder;

            pool.executeBlocking(
                    () -> IntStream.range(0, fsplits)
                            .parallel()
                            .forEach(i -> {
                                        try (Timer.Context c = timer.time()) {
                                            action.accept(offset + (i * fchunkSize), fremainder > 0 && i == fsplits - 1 ? fchunkSize + fremainder : fchunkSize);
                                            c.stop();
                                        }
                                    }
                            )
            );
        }
    }
}
