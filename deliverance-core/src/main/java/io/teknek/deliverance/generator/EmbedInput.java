package io.teknek.deliverance.generator;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.TensorShape;

public interface EmbedInput {
    AbstractTensor inputTokenToEmbedding(int inputToken, int position);

    default AbstractTensor batchInputsToEmbeddings(int[] inputTokens, int startPos) {
        Preconditions.checkArgument(inputTokens.length > 0);
        AbstractTensor t = inputTokenToEmbedding(inputTokens[0], startPos);
        if (inputTokens.length == 1) {
            return t;
        }
        TensorShape tbs = TensorShape.of(inputTokens.length, t.shape().last());
        AbstractTensor tb = TensorCache.instance.get(t.dType(), tbs);
        tb.copyFrom(t, 0, 0, t.shape().last());
        t.close();
        VectorMath.pfor(1, inputTokens.length, i -> {
            AbstractTensor ti = inputTokenToEmbedding(inputTokens[i], startPos + i);
            tb.copyFrom(ti, 0, i * ti.shape().last(), ti.shape().last());
            ti.close();
        });
        return tb;
    }
}

