package io.teknek.deliverance.tensor;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

public class AbstractTensorUtils {

    private static final Logger LOGGER = LoggerFactory.getLogger(AbstractTensorUtils.class.getName());

    /*
    public AbstractTensor quantize(DType dType) {
        return quantize(dType, false);
    }

    public AbstractTensor quantize(DType dType, boolean force) {
        if (!force && (this.shape().first() == 1 || this.dType == dType || this.dType.size() < dType.size())) {
            return this;
        }
        if (shape.isSparse()) {
            logger.info("Quantizing sparse tensor is not supported");
            return this;
        }
        return switch (dType) {
            case Q4 -> new Q4ByteBufferTensor(this);
            case I8 -> new Q8ByteBufferTensor(this);
            case F32 -> new FloatBufferTensor(this);
            case BF16 -> new BFloat16BufferTensor(this);
            default -> this;
        };
    }*/

    public static AbstractTensor quantize(AbstractTensor input, DType targetType, boolean force){
        if (!force && (input.shape().first() == 1 || input.dType == targetType || input.dType.size() < targetType.size())) {
            return input;
        }
        if (input.shape.isSparse()) {
            LOGGER.info("Quantizing sparse tensor is not supported");
            return input;
        }
        return switch (targetType) {
            case Q4 -> new Q4ByteBufferTensor(input);
            case I8 -> new Q8ByteBufferTensor(input);
            case F32 -> new FloatBufferTensor(input);
            case BF16 -> new BFloat16BufferTensor();
            default -> input;
        };
    }

    public static AbstractTensor quantize(AbstractTensor input, DType dType) {
        return quantize(input, dType, false);
    }
}
