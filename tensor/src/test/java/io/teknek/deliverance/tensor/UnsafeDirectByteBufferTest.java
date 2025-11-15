package io.teknek.deliverance.tensor;

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;

import static io.teknek.deliverance.tensor.UnsafeDirectByteBuffer.allocateAlignedByteBuffer;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class UnsafeDirectByteBufferTest {

    @Test
    void basic(){
        ByteBuffer bb = allocateAlignedByteBuffer(16, UnsafeDirectByteBuffer.CACHE_LINE_SIZE);
        bb.put( 0, (byte) 4);
        assertEquals((byte)4, bb.get(0));
    }

    @Test
    void badArg(){
        ByteBuffer bb = allocateAlignedByteBuffer(3, UnsafeDirectByteBuffer.CACHE_LINE_SIZE);

    }
}
