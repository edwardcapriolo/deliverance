package io.teknek.deliverance.model;

import com.codahale.metrics.Histogram;
import io.teknek.deliverance.CausualWhisperer;
import io.teknek.deliverance.generator.LayerNorm;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tokenizer.Tokenizer;
import net.jafama.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class GuidedChoiceSampler {
    private static final Logger LOG = LoggerFactory.getLogger(GuidedChoiceSampler.class);
    private final AbstractModel abstractModel;
    private final Histogram forward1;
    private final Histogram dotprod2;
    private final Histogram fullSample;

    private final AbstractTensor output;

    private final AbstractTensor logits;
    private final LayerNorm layerNorm;
    private final Tokenizer tokenizer;
    private final List<String> currentChoices;
    private final StringBuilder current;

    public GuidedChoiceSampler(AbstractModel abstractModel, AbstractTensor output,
                               AbstractTensor logits, LayerNorm layerNorm, Tokenizer tokenizer, List<String> choices, StringBuilder current) {
        this.abstractModel = abstractModel;
        this.output = output;
        this.logits = logits;
        this.layerNorm = layerNorm;
        this.tokenizer = tokenizer;
        this.currentChoices = choices;
        this.current = current;

        forward1 = abstractModel.metricRegistry.histogram("sample.foward1");
        dotprod2 = abstractModel.metricRegistry.histogram("sample.dotproduct2");
        fullSample = abstractModel.metricRegistry.histogram("sample.fullsample");

    }

    /**
     * By looking at the current response and sampling we attempt to find the next token that fits one of the choices
     * the user has requested
     * @return the next token that matches some of the guided input, if no match vocabularySize - 1 is returned
     */
    public int sample() {
        long start = System.nanoTime();
        try (AbstractTensor embedding = layerNorm.forward(output)) {
            long afterForward = System.nanoTime();
            forward1.update(Math.abs(afterForward - start));

            VectorMath.pchunk(0, abstractModel.config.vocabularySize, (chunkStart, chunkSize) -> {
                abstractModel.configurableTensorProvider.get()
                        .dotProductChunk(logits, embedding, abstractModel.sampleOutput.getOutputLogitsWeights(), 0,
                                abstractModel.config.embeddingLength, chunkStart, chunkSize);
            }, abstractModel.configurableTensorProvider.get().parallelSplitSize());
            long afterDotProductChunk = System.nanoTime();
            dotprod2.update(Math.abs(afterDotProductChunk - afterForward));

            if (abstractModel.config.logitMultiplier != null) {
                CausualWhisperer.LOGGER.debug("scaling logits logitMultiplier: {}", abstractModel.config.logitMultiplier);
                abstractModel.configurableTensorProvider.get().scale(1.0f / abstractModel.config.logitMultiplier,
                        logits, 0, abstractModel.config.vocabularySize);
            }
            int maxi = Integer.MIN_VALUE;
            double maxv = Double.NEGATIVE_INFINITY;
            String bestMatch = "";
            for (int i = 0; i < abstractModel.config.vocabularySize; i++) {
                float v = logits.get(0, i);
                if (abstractModel.config.finalLogitSoftCapping != null) {
                    v /= abstractModel.config.finalLogitSoftCapping;
                    v = (float) FastMath.tanh(v);
                    v = v * abstractModel.config.finalLogitSoftCapping;
                    logits.set(v, 0, i);
                }
                if (v > maxv) {
                    String decodedToken = tokenizer.decode(i);
                    String entire = current + decodedToken;
                    if (!decodedToken.isEmpty() && currentChoices.stream().anyMatch(ch -> ch.startsWith(entire))) {
                        if (LOG.isDebugEnabled()) {
                            LOG.debug("candidate found token {} {} {} ", i, decodedToken, entire);
                        }
                        if (entire.length() > bestMatch.length()) {
                            maxi = i;
                            maxv = v;
                            bestMatch = decodedToken;
                        }
                    }
                }
            }
            if (maxi != Integer.MIN_VALUE) {
                return maxi;
            }
            CausualWhisperer.LOGGER.debug("Reached end returning {}", abstractModel.config.vocabularySize - 1);
            return abstractModel.config.vocabularySize - 1;
        } finally {
            long end = System.nanoTime();
            fullSample.update(Math.abs(end - start));
        }
    }
}

/*
candidate found token 12662 Gi Gi
candidate found token 65060 Gian Gian
candidate found token 235319 G G
return Gian
candidate found token 617 ts Giants
return ts
Giants
Giants



candidate found token 6151 Je Je
candidate found token 65060 Gian Gian
candidate found token 235319 G G
candidate found token 235338 J J
return Gian
candidate found token 617 ts Giants
return ts
Giants
Giants
 */
