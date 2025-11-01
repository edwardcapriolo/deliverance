/*
 * Copyright 2024 Edward Guy Capriolo
 *
 * The Deliverance Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package io.teknek.deliverance.model.qwen2;


import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.CausalSelfAttention;
import io.teknek.deliverance.generator.MLPBlock;
import io.teknek.deliverance.generator.RmsNorm;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.model.llama.LlamaModel;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tokenizer.Tokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.stream.IntStream;

import static io.teknek.deliverance.tensor.AbstractTensorUtils.quantize;

public class Qwen2Model extends LlamaModel {

    private static final Logger logger = LoggerFactory.getLogger(Qwen2Model.class);

    public Qwen2Model(
            InferenceType inferenceType, Config c, WeightLoader w, Tokenizer t, DType workingMemoryDType,
            DType workingMemoryQType, Optional<DType> modelQType,
            ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
            TensorCache tensorCache, KvBufferCacheSettings kvBufferCacheSettings
    ) {
        super(inferenceType, c, w, t, workingMemoryDType, workingMemoryQType, modelQType, configurableTensorProvider, metricRegistry,
                tensorCache, kvBufferCacheSettings);

    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock[] transformerBlocks = new TransformerBlock[config.dctx().numberOfLayers];
        IntStream.range(config.dctx().layerStart, config.dctx().layerEnd).parallel().forEach(i -> {
            int relativeLayer = i - config.dctx().layerStart; // FIXME: add a helper to the context
            String base = "model.layers." + i + ".";
            String prefix = base + "self_attn.";
            CausalSelfAttention attention = new CausalSelfAttention(
                    this,
                    relativeLayer,
                    Optional.of(quantize(weights.load(prefix + "q_proj.bias"), qType)),
                    Optional.of(quantize(weights.load(prefix + "k_proj.bias"), qType)),
                    Optional.of(quantize(weights.load(prefix + "v_proj.bias"), qType)),
                    quantize(weights.load(prefix + "q_proj.weight", config.dctx(), true, false), qType),
                    quantize(weights.load(prefix + "k_proj.weight", config.dctx(), true, false), qType),
                    quantize(weights.load(prefix + "v_proj.weight", config.dctx(), true, false), qType),
                    Optional.empty(),
                    quantize(weights.load(prefix + "o_proj.weight", config.dctx(), false, true), qType),
                    configurableTensorProvider
            );

            prefix = base + "mlp.";
            MLPBlock mlp = new MLPBlock(
                    this,
                    config.activationFunction,
                    quantize(weights.load(prefix + "gate_proj.weight", config.dctx(), true, false), qType), // w1
                    quantize(weights.load(prefix + "down_proj.weight", config.dctx(), false, true), qType), // w2
                    quantize(weights.load(prefix + "up_proj.weight", config.dctx(), true, false), qType),
                    configurableTensorProvider
            ); // w3

            transformerBlocks[relativeLayer] = new TransformerBlock(
                    this,
                    relativeLayer,
                    new RmsNorm(this, quantize(weights.load(base + "input_layernorm.weight"), qType), metricRegistry),
                    attention,
                    new RmsNorm(this, quantize(weights.load(base + "post_attention_layernorm.weight"), qType), metricRegistry),
                    mlp,
                    configurableTensorProvider
            );
        });

        return transformerBlocks;
    }


    /*
    @Override
    public ModelSupport.ModelType getModelType() {
        return ModelSupport.getModelType("QWEN2");
    }

     */
}
