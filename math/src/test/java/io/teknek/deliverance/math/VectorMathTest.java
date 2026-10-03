package io.teknek.deliverance.math;

import org.junit.jupiter.api.Test;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.*;

/**
 * Unit tests for the {@code VectorMath} class, which provides methods for performing vectorized
 * operations using parallel processing. These tests cover various scenarios, including parallel
 * iteration, offset and uneven chunking, and per-chunk metrics timing.
 */
public class VectorMathTest {
    public WrappedForkJoinPool getPool(){
        return new WrappedForkJoinPool(new ForkJoinPool(1));
    }

    @Test
    void testPchunk(){
        try ( WrappedForkJoinPool underlying = getPool()) {
            BiIntConsumer b = mock(BiIntConsumer.class);
            VectorMath.pchunk(0, 10, b, 2, underlying);
            verify(b).accept(0, 5);
            verify(b).accept(5, 5);
            verifyNoMoreInteractions(b);
        }
    }

    @Test
    void testPchunkUneven(){
        try ( WrappedForkJoinPool underlying = getPool()) {
            BiIntConsumer b = mock(BiIntConsumer.class);
            VectorMath.pchunk(0, 9, b, 2, underlying);
            verify(b).accept(0, 4);
            verify(b).accept(4, 5);
            verifyNoMoreInteractions(b);
        }
    }

    @Test
    void testPchunkAgain() {
        try (WrappedForkJoinPool underlying = getPool()) {
            BiIntConsumer b = mock(BiIntConsumer.class);
            VectorMath.pchunk(0, 10, b, 5, underlying);
            verify(b).accept(0, 2);
            verify(b).accept(2, 2);
            verify(b).accept(4, 2);
            verify(b).accept(6, 2);
            verify(b).accept(8, 2);
            verifyNoMoreInteractions(b);
        }
    }

    @Test
    void pforProcessesEveryValueInTheRequestedRange() {
        try (WrappedForkJoinPool underlying = getPool()) {
            Set<Integer> values = ConcurrentHashMap.newKeySet();

            VectorMath.pfor(3, 8, values::add, underlying);

            assertEquals(Set.of(3, 4, 5, 6, 7), values);
        }
    }

    @Test
    void pchunkPreservesOffsetAndAssignsRemainderToTheLastChunk() {
        try (WrappedForkJoinPool underlying = getPool()) {
            Set<String> chunks = ConcurrentHashMap.newKeySet();

            VectorMath.pchunk(2, 10, (offset, length) -> chunks.add(offset + ":" + length), 3, underlying);

            assertEquals(Set.of("2:3", "5:3", "8:4"), chunks);
        }
    }

    @Test
    void pchunkUsesOneDirectChunkWhenSplitSizeExceedsLength() {
        try (WrappedForkJoinPool underlying = getPool()) {
            BiIntConsumer action = mock(BiIntConsumer.class);

            VectorMath.pchunk(7, 1, action, 10, underlying);

            verify(action).accept(7, 1);
            verifyNoMoreInteractions(action);
        }
    }
}
