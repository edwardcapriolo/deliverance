package io.teknek.deliverance.model;

import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.VectorTensorMathUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeSet;

/**
 *       //abel='xtc_threshold', info='If 2 or more tokens have probability above this threshold, consider removing all but the last one.')
 *         //label='xtc_probability', info='Probability that the removal will actually happen. 0 disables the sampler. 1 makes it always happen.')
 *
 *         float xtcThreshold = 0.1f;
 *         float xtcProbability = 0.5f;
 */
public class ExcludeTopChoicePicker {

    private static final Logger LOGGER =  LoggerFactory.getLogger(ExcludeTopChoicePicker.class);
    private final AbstractTensor logits;
    private final float xtcThreshold;
    private final float xtcProbabilty;
    private final AbstractModel abstractModel;
    private final Random random;

    public ExcludeTopChoicePicker(AbstractModel abstractModel, AbstractTensor logits, float xtcThreshold,
                                  float xtcProbabilty, Random random) {
        this.logits = logits;
        this.xtcThreshold = xtcThreshold;
        this.xtcProbabilty = xtcProbabilty;
        this.abstractModel = abstractModel;
        this.random = random;
    }

    /**
     *
     * @return Some if this sampler found a candidate token non if the not activated
     */
    public Optional<IndexValueToken> process(){
        float pick = random.nextFloat();
        if (pick > xtcProbabilty){
            return Optional.empty();
        }
        SortedSet<IndexValueToken> aboveThreshold = new TreeSet<>();
        AbstractTensor logSum = abstractModel.getTensorAllocator().getDirty(logits.dType(), logits.shape());
        VectorTensorMathUtils.logSumExpTensor(logSum, logits);
        for (int i =0; i< logSum.size(); i++) {
            float ls = logSum.get(0, i);
            float prob = (float) Math.exp(ls);

                if (prob > xtcThreshold){
                PreTrainedTokenizer tokenizer = abstractModel.getPreTrainedTokenizer();
                if (tokenizer == null){
                    throw new IllegalStateException("tokenizer is null");
                }
                IndexValueToken token = new IndexValueToken(i, logits.get(0, i), abstractModel.decodeToken(i));
                token.logProb = ls;
                if (aboveThreshold.isEmpty() || aboveThreshold.size() == 1){
                    aboveThreshold.add(token);
                } else {
                    if (token.compareTo(aboveThreshold.first()) < 0){
                        aboveThreshold.removeFirst();
                        aboveThreshold.add(token);
                    }
                    if (token.compareTo(aboveThreshold.last()) > 0){
                        aboveThreshold.removeLast();
                        aboveThreshold.add(token);
                    }
                }
            }
        }
        if (aboveThreshold.isEmpty()){
            return Optional.empty();
        }
        if (abstractModel.getPreTrainedTokenizer().allSpecialIds().contains(aboveThreshold.last().index)){
            if (LOGGER.isDebugEnabled()){
                LOGGER.debug("Special token {} at max index.", aboveThreshold.last());
            }
            return Optional.of(aboveThreshold.last());
        }
        if (LOGGER.isDebugEnabled()){
            LOGGER.debug("From the list: {} selected: {}", aboveThreshold, aboveThreshold.last());
        }
        return Optional.of(aboveThreshold.first());

    }
}
