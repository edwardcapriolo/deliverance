package io.teknek.deliverance.model.mistral;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.TokenRenderer;
import io.teknek.deliverance.model.llama.LlamaModel;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tokenizer.Tokenizer;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.util.Optional;

public class MistralModel extends LlamaModel {

    public MistralModel(InferenceType inferenceType, Config c, WeightLoader w, Tokenizer t, DType workingMemoryDType,
                      DType workingMemoryQType, Optional<DType> modelQType,
                      ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
                      TensorCache tensorCache, KvBufferCacheSettings kvBufferCacheSettings, TokenRenderer tokenRenderer,
                        ToolCallParser toolCallParser) {
        super(inferenceType, c, w, t, workingMemoryDType, workingMemoryQType, modelQType, configurableTensorProvider,
                metricRegistry, tensorCache, kvBufferCacheSettings, tokenRenderer, toolCallParser);
    }
}
