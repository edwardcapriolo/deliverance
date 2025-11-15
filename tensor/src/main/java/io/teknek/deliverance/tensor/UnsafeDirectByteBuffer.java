package io.teknek.deliverance.tensor;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class UnsafeDirectByteBuffer {

    private static VarHandle VAR_HANDLE_ADDRESS;
    public static final int CACHE_LINE_SIZE = 64;

    static {
        try {
            VAR_HANDLE_ADDRESS = MethodHandles
                    .privateLookupIn(ByteBuffer.class, MethodHandles.lookup()).findVarHandle(ByteBuffer.class,
                            "address", long.class);
        } catch (RuntimeException | IllegalAccessException | NoSuchFieldException e) {
            throw new RuntimeException(e);
        }
    }

    public static long getAddress(ByteBuffer buffy) {
        if (VAR_HANDLE_ADDRESS != null) {
            return (Long) VAR_HANDLE_ADDRESS.get(buffy);
        }
        throw new IllegalArgumentException("unreachable");
    }

    public static ByteBuffer allocateAlignedByteBuffer(int capacity, long align) {
        if (Long.bitCount(align) != 1) {
            throw new IllegalArgumentException("Alignment must be a power of 2");
        }
        // We over allocate by the alignment so we know we can have a large
        // enough aligned block of memory to use.
        ByteBuffer buffy = ByteBuffer.allocateDirect((int) (capacity + align));
        long address = getAddress(buffy);
        if ((address & (align - 1)) == 0) {
            // limit to the capacity specified
            buffy.limit(capacity);
            // set order to native while we are here.
            ByteBuffer slice = buffy.slice().order(ByteOrder.nativeOrder());
            // the slice is now an aligned buffer of the required capacity
            return slice;
        } else {
            int newPosition = (int) (align - (address & (align - 1)));
            buffy.position(newPosition);
            int newLimit = newPosition + capacity;
            // limit to the capacity specified
            buffy.limit(newLimit);
            // set order to native while we are here.
            ByteBuffer slice = buffy.slice().order(ByteOrder.nativeOrder());
            // the slice is now an aligned buffer of the required capacity
            return slice;
        }
    }
}