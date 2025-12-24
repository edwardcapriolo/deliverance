package io.teknek.deliverance.math;

import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.Mockito;

import static org.mockito.Mockito.*;

public class VectorMathTest {
    @Test
    void testPchunk(){
        BiIntConsumer b = mock(BiIntConsumer.class);
        VectorMath.pchunk(0, 10, b, 2);
        verify(b).accept(0,5);
        verify(b).accept(5,5);
        verifyNoMoreInteractions(b);
    }

    @Test
    void testPchunkUneven(){
        BiIntConsumer b = mock(BiIntConsumer.class);
        VectorMath.pchunk(0, 9, b, 2);
        verify(b).accept(0,4);
        verify(b).accept(4,5);
        verifyNoMoreInteractions(b);
    }

    @Test
    void testPchunkAgain(){
        BiIntConsumer b = mock(BiIntConsumer.class);
        VectorMath.pchunk(0, 10, b, 5);
        verify(b).accept(0,2);
        verify(b).accept(2,2);
        verify(b).accept(4,2);
        verify(b).accept(6,2);
        verify(b).accept(8,2);
        verifyNoMoreInteractions(b);
    }

}
