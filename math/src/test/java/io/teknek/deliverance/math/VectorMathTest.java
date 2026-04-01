package io.teknek.deliverance.math;

import org.junit.jupiter.api.Test;

import java.util.concurrent.ForkJoinPool;

import static org.mockito.Mockito.*;

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

}
