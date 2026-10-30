# Gemma4 Decoder Layer: Side By Side

This is a direct comparison of the current source. It is not a claim of parity.

Sources:

- Transformers: `/ai-code/transformers/src/transformers/models/gemma4/modular_gemma4.py:1192-1249`
- Deliverance: `/ai-code/del5/core/src/main/java/io/teknek/deliverance/generator2/TransformerBlock2.java:126-224`

<table>
<tr>
<th>Transformers Gemma4TextDecoderLayer.forward</th>
<th>Deliverance TransformerBlock2.forward</th>
</tr>
<tr>
<td valign="top">

<pre>
residual = hidden_states

hidden_states = self.input_layernorm(hidden_states)
hidden_states, _ = self.self_attn(
    hidden_states=hidden_states,
    position_embeddings=position_embeddings,
    attention_mask=attention_mask,
    shared_kv_states=shared_kv_states,
    position_ids=position_ids,
    past_key_values=past_key_values,
    **kwargs,
)
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = residual + hidden_states

residual = hidden_states
hidden_states = self.pre_feedforward_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)

if self.enable_moe_block:
    hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

    hidden_states_flat = residual.reshape(-1, residual.shape[-1])
    _, top_k_weights, top_k_index = self.router(hidden_states_flat)
    hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
    hidden_states_2 = self.experts(
        hidden_states_2, top_k_index, top_k_weights
    )
    hidden_states_2 = hidden_states_2.reshape(residual.shape)
    hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

    hidden_states = hidden_states_1 + hidden_states_2

hidden_states = self.post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states

if self.hidden_size_per_layer_input:
    residual = hidden_states
    hidden_states = self.per_layer_input_gate(hidden_states)
    hidden_states = self.act_fn(hidden_states)
    hidden_states = hidden_states * per_layer_input
    hidden_states = self.per_layer_projection(hidden_states)
    hidden_states = self.post_per_layer_input_norm(hidden_states)
    hidden_states = residual + hidden_states

hidden_states = apply_layer_scalar(hidden_states)
return hidden_states
</pre>

</td>
<td valign="top">

<pre>
TensorRef residual = embedding;
TensorRef preAttentionHiddenStates = preAttentionNorm
        .map(ln -&gt; ln.forward(embedding))
        .orElse(embedding);

TensorRef postAttention;
try (AbstractModel.TensorRefLease preAttentionInput =
        preAttentionProjectionInput(preAttentionHiddenStates)) {
    postAttention = attention.forward(
        preAttentionInput.tensor(), startPosition, kvSession,
        tensorReducer, phase, batchSize,
        sequenceLength, attentionMask);
}

hiddenStates = maybeApplyNorm(
        postAttention, postAttentionNorm);
applyResidual(hiddenStates, residual,
        "post_attention_residual");

TensorRef preFeedForward = preFFNorm
        .map(ln -&gt; ln.forward(hiddenStates))
        .orElse(hiddenStates);
TensorRef postFeedForward;
try (AbstractModel.TensorRefLease preFeedForwardInput =
        preFFProjectionInput(preFeedForward)) {
    postFeedForward = ffBlock.forward(
        preFeedForwardInput.tensor(), tensorReducer, phase);
}
TensorRef postFeedForwardNorm;

if (moeBlock != null &amp;&amp; moeDenseNorm.isPresent()
        &amp;&amp; moeInputNorm.isPresent()
        &amp;&amp; moeOutputNorm.isPresent()) {
    TensorRef denseBranch = moeDenseNorm
            .get().forward(postFeedForward);
    TensorRef expertInput = moeInputNorm
            .get().forward(hiddenStates);
    TensorRef expertOutput = moeBlock.forward(
            expertInput, tensorReducer, phase);
    TensorRef expertBranch = moeOutputNorm
            .get().forward(expertOutput);
    compositeOps.addResidual(
            denseBranch, expertBranch, 1.0f);
    postFeedForwardNorm = maybeApplyNorm(
            denseBranch, postFFNorm);
} else {
    postFeedForwardNorm = maybeApplyNorm(postFeedForward, postFFNorm);
}

applyResidual(postFeedForwardNorm, hiddenStates,
        "post_ff_residual");

if (perLayerInput != null
        &amp;&amp; pleGateWeights != null
        &amp;&amp; pleProjectionWeights != null
        &amp;&amp; pleNorm.isPresent()) {
    TensorRef pleGate = makeDenseTensorRef(
            batchSize, pleLength);
    lighter.dotProductRows(
            pleGate, postFeedForwardNorm, pleGateWeights, ...);

    TensorRef pleGated = makeDenseTensorRef(
            batchSize, pleLength);
    for (int row = 0; row &lt; batchSize; row++) {
        for (int column = 0; column &lt; pleLength; column++) {
            pleGated.set(
                activation(pleGate.get(row, column))
                    * perLayerInput.get(row, column),
                row, column);
        }
    }

    TensorRef pleProjected = makeDenseTensorRef(
            batchSize, embeddingLength);
    lighter.dotProductRows(
            pleProjected, gatedQ.tensor(),
            pleProjectionWeights, ...);
    TensorRef pleNormalized = pleNorm.get()
            .forward(pleProjected);
    compositeOps.addResidual(
            pleNormalized, postFeedForwardNorm, 1.0f);
    postFeedForwardNorm = pleNormalized;
}

postFeedForwardNorm = applyLayerScalar(postFeedForwardNorm);
return maybeApplyNorm(postFeedForwardNorm, preResponseNorm);
</pre>

</td>
</tr>
</table>

The Transformers source writes the final line inline as
`hidden_states *= self.layer_scalar`; `apply_layer_scalar` above is a comparison-only
name so the semantic operation aligns visually with the Java method call. The Java
helper retains its `layerScalar == 1.0` fast path.

## Known Differences

- Deliverance still has scalar Java loops for PLE activation/multiplication.
- Deliverance passes through generic optional norms and `preResponseNorm`, which must be proven equivalent to Gemma4's exact block fields.
- MoE input, routing, packed experts, and branch norm ordering require tensor-level parity, not structural similarity.
- KV sharing and heterogeneous per-layer KV widths occur inside the attention/cache implementation and are not visible in this block-level comparison.
- `...` above represents omitted Java argument plumbing only; it is not evidence that the omitted arguments are equivalent.

## Required Proof

Compare both implementations at these checkpoints for the same real checkpoint and input IDs:

```text
input embedding
PLE identity
PLE projection before/after norm
attention Q/K/V projection
Q/K/V normalization
RoPE output
attention output
dense MLP output
MoE router/top-k/expert output
post-FF residual
post-PLE output
layer-scalar output
final hidden state
pre-sampling logits
```
