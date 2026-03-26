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

    //Come up with a design that is more clear with the return value is it the input or a new tensor
    @Deprecated(forRemoval = true)
    public static AbstractTensor quantize(AbstractTensor input, DType targetType, boolean force){
        if (!force && (input.shape().first() == 1 || input.dType == targetType || input.dType.size() < targetType.size())) {
            return input;
        }
        if (input.shape.isSparse()) {
            LOGGER.warn("Quantizing sparse tensor is not supported");
            return input;
        }
        return switch (targetType) {
            case Q4 -> new Q4ByteBufferTensor(input);
            case I8 -> new Q8ByteBufferTensor(input);
            case F32 -> new FloatBufferTensor(input);
            case BF16 -> new BFloat16BufferTensor(input);
            default -> input;
        };
    }


    /**
     *
     * @param input a tensor to consider for input
     * @param dType the target type
     * @return A new tensor which has been quantized to the requested type, or the original tensor if the value can not
     * be quantized
     */
    public static AbstractTensor quantize(AbstractTensor input, DType dType) {
        return quantize(input, dType, false);
    }
}
