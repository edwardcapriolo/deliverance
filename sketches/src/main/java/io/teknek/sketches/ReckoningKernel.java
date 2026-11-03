package io.teknek.sketches;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.AbstractTensor;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor.impl.FloatBufferTensor;

//https://github.com/dottxt-ai/outlines-core/blob/main/outlines_core/kernels/numpy.py
public class ReckoningKernel {

    public AbstractTensor allocateTokenBitmask(int vocabSize){
        AbstractTensor tensor = new FloatBufferTensor(TensorShape.of(1, (vocabSize + 31 ) / 32));
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
