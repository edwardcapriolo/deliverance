package io.teknek.sketches;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;

//https://github.com/dottxt-ai/outlines-core/blob/main/outlines_core/kernels/numpy.py
public class DeliveranceKernel {

    public AbstractTensor allocateTokenBitmask(int vocabSize){
        ArrayQueueTensorAllocator c = new ArrayQueueTensorAllocator(new MetricRegistry());
        AbstractTensor tensor = c.getDirty(DType.F32, TensorShape.of(1, (vocabSize + 31 ) / 32));
        for (int i = 0; i < 3; i++ ){
            tensor.set(-1, new int [] { 0, i});
        }
        return tensor;
    }

/*
    def allocate_token_bitmask(vocab_size: int) -> np.ndarray:
            return np.full(
            (1, (vocab_size + 31) // 32),
            -1,
    dtype=np.int32,
            )*/
    public void applyTokenBitmaskInplace(AbstractTensor logics, AbstractTensor mask){

    }
    //def _apply_token_bitmask_inplace_kernel(logits, mask):
}
