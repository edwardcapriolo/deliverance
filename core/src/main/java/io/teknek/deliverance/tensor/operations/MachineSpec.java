package io.teknek.deliverance.tensor.operations;

import jdk.incubator.vector.FloatVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteOrder;

public class MachineSpec {
    private static final Logger logger = LoggerFactory.getLogger(MachineSpec.class);
    public static final Type VECTOR_TYPE = new MachineSpec().type;

    public enum Type {
        AVX_256(2),
        AVX_512(4),
        ARM_128(8),
        AVX_128(16),
        NONE(0);

        public final int ctag;

        Type(int cflag) {
            this.ctag = cflag;
        }
    }

    private final Type type;

    private MachineSpec() {
        Type tmp = Type.NONE;
        try {
            int preferredBits = FloatVector.SPECIES_PREFERRED.vectorBitSize();
            if (preferredBits == 512) {
                tmp = Type.AVX_512;
            } else if (preferredBits == 256) {
                tmp = Type.AVX_256;
            } else if (preferredBits == 128) {
                if (RuntimeSupport.isArm()) {
                    tmp = Type.ARM_128;
                } else {
                    tmp = Type.AVX_128;
                }
            }
            if (tmp == Type.NONE) {
                logger.warn("Unknown vector type: {}", preferredBits);
            }

        } catch (Throwable t) {
            logger.warn("Java SIMD Vector API *not* available. Add --add-modules=jdk.incubator.vector to your JVM options");
        }

        logger.debug("Machine Vector Spec: {}", tmp);
        logger.debug("Byte Order: {}", ByteOrder.nativeOrder().toString());
        type = tmp;
    }
}