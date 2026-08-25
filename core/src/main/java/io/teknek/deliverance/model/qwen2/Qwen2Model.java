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


import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.CausalSelfAttention;
import io.teknek.deliverance.generator.MLPBlock;
import io.teknek.deliverance.generator.RmsNorm;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.llama.LlamaModel;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.stream.IntStream;

import static io.teknek.deliverance.tensor.AbstractTensorUtils.quantize;

public class Qwen2Model extends LlamaModel {

    private static final Logger logger = LoggerFactory.getLogger(Qwen2Model.class);

    public Qwen2Model(
            InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer t, DType workingMemoryDType,
            DType workingMemoryQType, Optional<DType> modelQType,
            ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
            TensorAllocator arrayQueueTensorAllocator, KvBufferCacheSettings kvBufferCacheSettings,
            ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives,
            Optional<DType> outputHeadQuantization
    ) {
        super(inferenceType, c, w, t, workingMemoryDType, workingMemoryQType, modelQType, configurableTensorProvider, metricRegistry,
                arrayQueueTensorAllocator, kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext,
                tensorParallelCollectives, outputHeadQuantization);

    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock[] transformerBlocks = new TransformerBlock[config.numberOfLayers];
        IntStream.range(0, config.numberOfLayers).parallel().forEach(i -> {
            int relativeLayer = i;
            String base = "model.layers." + i + ".";
            String prefix = base + "self_attn.";
            String qName = prefix + "q_proj.weight";
            String kName = prefix + "k_proj.weight";
            String vName = prefix + "v_proj.weight";
            String oName = prefix + "o_proj.weight";
            CausalSelfAttention attention = new CausalSelfAttention(
                    this,
                    relativeLayer,
                    Optional.of(quantize(weights.load(prefix + "q_proj.bias"), qType)),
                    Optional.of(quantize(weights.load(prefix + "k_proj.bias"), qType)),
                    Optional.of(quantize(weights.load(prefix + "v_proj.bias"), qType)),
                    quantize(weights.load(qName), qType),
                    quantize(weights.load(kName), qType),
                    quantize(weights.load(vName), qType),
                    Optional.empty(),
                    quantize(weights.load(oName), qType),
                    configurableTensorProvider,
                    metricRegistry,
                    qName, kName, vName, oName
            );

            prefix = base + "mlp.";
            String gateName = prefix + "gate_proj.weight";
            String downName = prefix + "down_proj.weight";
            String upName = prefix + "up_proj.weight";
            MLPBlock mlp = new MLPBlock(
                    this,
                    config.activationFunction,
                    quantize(weights.load(gateName), qType), // w1
                    quantize(weights.load(downName), qType), // w2
                    quantize(weights.load(upName), qType),
                    configurableTensorProvider,
                    gateName, upName, downName
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

}
