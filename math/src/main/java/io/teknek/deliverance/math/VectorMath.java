package io.teknek.deliverance.math;

import java.util.function.IntConsumer;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VectorMath {

    private static final Logger logger = LoggerFactory.getLogger(VectorMath.class);

    public static void pfor(int start, int end, IntConsumer action) {
        PhysicalCoreTuningExecutor.instance.get().execute(() -> IntStream.range(start, end).parallel().forEach(action));
    }

    /**
     *
     * @param offset a starting offset
     * @param length a length of items to split starting from the offset
     * @param action an action to perform on each split
     * @param splitSize split the list into this many parts
     */
    public static void pchunk(int offset, int length, BiIntConsumer action, int splitSize) {
        int splits = Math.min(length, splitSize);
        int chunkSize = length / splits;

        if (splits == 1) {
            action.accept(offset, length);
        } else {
            int remainder = length % chunkSize;

            int fsplits = splits;
            int fchunkSize = chunkSize;
            int fremainder = remainder;

            PhysicalCoreTuningExecutor.instance.get()
                    .execute(
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

}
