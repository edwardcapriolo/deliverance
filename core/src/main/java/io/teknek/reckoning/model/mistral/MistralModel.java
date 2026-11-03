package io.teknek.reckoning.model.mistral;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.grace.PreTrainedTokenizer;
import io.teknek.reckoning.math.WrappedForkJoinPool;
import io.teknek.reckoning.model.llama.LlamaModel;
import io.teknek.reckoning.model.tensorparallel.TensorParallelCollectives;
import io.teknek.reckoning.model.tensorparallel.TensorParallelContext;
import io.teknek.reckoning.safetensors.Config;
import io.teknek.reckoning.safetensors.WeightLoader;
import io.teknek.reckoning.tensor.KvBufferCacheSettings;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.toolcallparser.ToolCallParser;

import java.util.Optional;

public class MistralModel extends LlamaModel {

    public MistralModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer t,
                        DType workingMemoryDType, DType workingMemoryQType, Optional<DType> modelQType,
                        MetricRegistry metricRegistry, KvBufferCacheSettings kvBufferCacheSettings,
                        ToolCallParser toolCallParser, WrappedForkJoinPool pool,
                        TensorParallelContext tensorParallelContext, TensorParallelCollectives tensorParallelCollectives,
                        Optional<DType> outputHeadQuantization, Lighter lighter) {
        super(inferenceType, c, w, t, workingMemoryDType, workingMemoryQType, modelQType, metricRegistry,
                kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext, tensorParallelCollectives,
                outputHeadQuantization, lighter);
    }

    @Override
    protected boolean usesTensorRefExecution() {
        return true;
    }

    @Override
    public boolean usesKvCache2Generation() {
        return true;
    }
}
