package io.teknek.deliverance.generator;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class EmbedInput {
    public static final Logger LOGGER = LoggerFactory.getLogger(EmbedInput.class);

    protected AbstractModel parent;

    public EmbedInput(AbstractModel parent){
        this.parent = parent;
    }

    public abstract AbstractTensor inputTokenToEmbedding(int inputToken, int position);

    public AbstractTensor batchInputsToEmbeddings(int[] inputTokens, int startPos) {
        Preconditions.checkArgument(inputTokens.length > 0);
        AbstractTensor zeroTokenEmbedding = inputTokenToEmbedding(inputTokens[0], startPos);

        LOGGER.warn("tensor for 0th inputToken shape {} size {}", zeroTokenEmbedding.shape(), zeroTokenEmbedding.size());
        if (inputTokens.length == 1) {
            return zeroTokenEmbedding;
        }
        TensorShape embeddingsFoEachInputToken = TensorShape.of(inputTokens.length, zeroTokenEmbedding.shape().last());
        AbstractTensor tb = parent.getTensorCache().get(zeroTokenEmbedding.dType(), embeddingsFoEachInputToken);
        tb.copyFrom(zeroTokenEmbedding, 0, 0, zeroTokenEmbedding.shape().last());
        zeroTokenEmbedding.close();
        VectorMath.pfor(1, inputTokens.length, i -> {
            AbstractTensor ti = inputTokenToEmbedding(inputTokens[i], startPos + i);
            tb.copyFrom(ti, 0, i * ti.shape().last(), ti.shape().last());
            ti.close();
        });
        return tb;
    }
}

