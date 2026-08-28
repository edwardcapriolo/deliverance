package io.teknek.deliverance.model.diffusiongemma;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;

/** Source-aligned mask construction helpers for DiffusionGemma. */
public final class DiffusionGemmaMaskSupport {
    private DiffusionGemmaMaskSupport() {
    }

    /** Minimal cache metadata needed by DiffusionGemma decoder mask construction. */
    public record CacheState(boolean compileable, int validCacheTokens, int maxCacheLength) {
        public static CacheState dynamic(int validCacheTokens) {
            return new CacheState(false, validCacheTokens, validCacheTokens);
        }

        public static CacheState statik(int validCacheTokens, int maxCacheLength) {
            return new CacheState(true, validCacheTokens, maxCacheLength);
        }

        public CacheState {
            Preconditions.checkArgument(validCacheTokens >= 0, "validCacheTokens must be >= 0");
            Preconditions.checkArgument(maxCacheLength >= validCacheTokens,
                    "maxCacheLength must be >= validCacheTokens");
        }
    }

    /** Full and sliding decoder attention masks. Either tensor may be {@code null} for HF's dynamic/no-padding shortcut. */
    public record DecoderAttentionMask(AbstractTensor fullAttention, AbstractTensor slidingAttention)
            implements AutoCloseable {
        @Override
        public void close() {
            if (fullAttention != null) {
                fullAttention.close();
            }
            if (slidingAttention != null && slidingAttention != fullAttention) {
                slidingAttention.close();
            }
        }
    }

    /**
     * Creates DiffusionGemma decoder masks following HF `create_diffusion_decoder_attention_mask` semantics.
     *
     * <p>The mask is bidirectional over the current canvas. The cache prefix comes from {@code decoderAttentionMask},
     * where {@code 1.0} means visible and {@code 0.0} means padding/unfilled cache. For dynamic cache without padding, HF
     * returns {@code None} masks and lets the attention implementation use its all-visible fast path; this method mirrors
     * that by returning null tensors.</p>
     */
    public static DecoderAttentionMask createDiffusionDecoderAttentionMask(DiffusionGemmaTextConfig config,
            int batchSize, int canvasLength, CacheState cacheState, AbstractTensor decoderAttentionMask) {
        Preconditions.checkArgument(config != null, "config must not be null");
        Preconditions.checkArgument(batchSize > 0, "batchSize must be > 0");
        Preconditions.checkArgument(canvasLength > 0, "canvasLength must be > 0");
        if (cacheState == null) {
            throw new IllegalArgumentException("cacheState is required for DiffusionGemma decoder masks");
        }
        if (cacheState.compileable() && decoderAttentionMask == null) {
            throw new IllegalArgumentException("static cache requires decoderAttentionMask");
        }
        if (decoderAttentionMask == null || (!cacheState.compileable() && allVisible(decoderAttentionMask))) {
            return new DecoderAttentionMask(null, null);
        }

        int fullCacheKvLength = cacheState.compileable() ? cacheState.maxCacheLength() : cacheState.validCacheTokens();
        int fullKvLength = fullCacheKvLength + canvasLength;
        validateDecoderAttentionMask(decoderAttentionMask, batchSize, fullKvLength);
        validateVisibleCounts(decoderAttentionMask, cacheState.validCacheTokens() + canvasLength);

        FloatBufferTensor full = new FloatBufferTensor(batchSize, 1, canvasLength, fullKvLength);
        for (int batch = 0; batch < batchSize; batch++) {
            for (int query = 0; query < canvasLength; query++) {
                for (int key = 0; key < fullKvLength; key++) {
                    full.set(decoderAttentionMask.get(batch, key), batch, 0, query, key);
                }
            }
        }

        int validCacheTokens = cacheState.validCacheTokens();
        int slidingStart;
        int slidingEnd;
        if (validCacheTokens >= config.slidingWindow) {
            slidingStart = cacheState.compileable()
                    ? validCacheTokens - config.slidingWindow
                    : validCacheTokens - config.slidingWindow + 1;
            slidingEnd = validCacheTokens;
        } else {
            slidingStart = 0;
            slidingEnd = cacheState.compileable()
                    ? Math.min(config.slidingWindow, cacheState.maxCacheLength())
                    : validCacheTokens;
        }
        int slidingCacheLength = slidingEnd - slidingStart;
        FloatBufferTensor sliding = new FloatBufferTensor(batchSize, 1, canvasLength,
                slidingCacheLength + canvasLength);
        for (int batch = 0; batch < batchSize; batch++) {
            for (int query = 0; query < canvasLength; query++) {
                for (int key = 0; key < slidingCacheLength; key++) {
                    sliding.set(decoderAttentionMask.get(batch, slidingStart + key), batch, 0, query, key);
                }
                for (int key = slidingCacheLength; key < slidingCacheLength + canvasLength; key++) {
                    sliding.set(1.0f, batch, 0, query, key);
                }
            }
        }

        return new DecoderAttentionMask(full, sliding);
    }

    private static boolean allVisible(AbstractTensor mask) {
        validateMaskTensor(mask);
        for (int batch = 0; batch < mask.shape().first(); batch++) {
            for (int key = 0; key < mask.shape().last(); key++) {
                if (mask.get(batch, key) == 0.0f) {
                    return false;
                }
            }
        }
        return true;
    }

    private static void validateDecoderAttentionMask(AbstractTensor mask, int batchSize, int fullKvLength) {
        validateMaskTensor(mask);
        Preconditions.checkArgument(mask.shape().first() == batchSize && mask.shape().last() == fullKvLength,
                "decoderAttentionMask must have shape [batchSize, cacheLength + canvasLength]");
    }

    private static void validateMaskTensor(AbstractTensor mask) {
        Preconditions.checkArgument(mask != null, "decoderAttentionMask must not be null");
        Preconditions.checkArgument(mask.dims() == 2, "decoderAttentionMask must be 2D");
        Preconditions.checkArgument(mask.dType() == DType.F32, "decoderAttentionMask must be F32 1/0 tensor");
    }

    private static void validateVisibleCounts(AbstractTensor mask, int maxVisibleTokens) {
        for (int batch = 0; batch < mask.shape().first(); batch++) {
            int visible = 0;
            for (int key = 0; key < mask.shape().last(); key++) {
                if (mask.get(batch, key) != 0.0f) {
                    visible++;
                }
            }
            if (visible > maxVisibleTokens) {
                throw new IllegalArgumentException("decoderAttentionMask has more visible positions than cached + canvas tokens");
            }
        }
    }
}
