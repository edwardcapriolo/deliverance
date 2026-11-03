package io.teknek.deliverance.model.bert;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.util.Optional;

/** XLM-R model identity over the shared RoBERTa implementation. */
public class XlmRobertaModel extends RobertaModel {
    public XlmRobertaModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer tokenizer,
            DType workingDType, DType workingQType, Optional<DType> modelQType,
            MetricRegistry metricRegistry, KvBufferCacheSettings kvBufferCacheSettings,
            ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization, Lighter lighter) {
        super(inferenceType, c, w, tokenizer, workingDType, workingQType, modelQType,
                metricRegistry, kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext,
                tensorParallelCollectives, outputHeadQuantization, lighter);
    }
}
