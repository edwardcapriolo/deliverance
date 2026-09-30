package io.teknek.deliverance.generator;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class EmbedInput {
    public static final Logger LOGGER = LoggerFactory.getLogger(EmbedInput.class);

    protected AbstractModel parent;

    public EmbedInput(AbstractModel parent){
        this.parent = parent;
    }

    public abstract TensorRef inputTokenToEmbedding(int inputToken, int position);

    public TensorRef batchInputsToEmbeddings(int[] inputTokens, int startPos) {
        try (Timer.Context ignored = InferenceProfiler.timer(parent.getMetricRegistry(), "embedinput.batch_inputs").time()) {
        Preconditions.checkArgument(inputTokens.length > 0);
        TensorRef zeroTokenEmbedding = inputTokenToEmbedding(inputTokens[0], startPos);

        LOGGER.debug("tensor for 0th inputToken shape {} size {}", zeroTokenEmbedding.shape(),
                zeroTokenEmbedding.shape().size());
        if (inputTokens.length == 1) {
            return zeroTokenEmbedding;
        }
        Lighter lighter = parent.getLighter();
        TensorRef tb = lighter.allocate(zeroTokenEmbedding.dType(),
                io.teknek.deliverance.tensor.TensorShape.of(inputTokens.length, zeroTokenEmbedding.shape().last()));
        lighter.copy(zeroTokenEmbedding, 0, tb, 0, (int) zeroTokenEmbedding.shape().last());
        zeroTokenEmbedding.close();
        for (int i = 1; i < inputTokens.length; i++) {
            TensorRef ti = inputTokenToEmbedding(inputTokens[i], startPos + i);
            lighter.copy(ti, 0, tb, i * (int) ti.shape().last(), (int) ti.shape().last());
            ti.close();
        }
        return tb;
        }
    }
}
