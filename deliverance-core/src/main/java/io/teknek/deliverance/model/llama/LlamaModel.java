package io.teknek.deliverance.model.llama;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tokenizer.Tokenizer;

import java.util.Optional;

public class LlamaModel extends AbstractModel {

    public LlamaModel(InferenceType inferenceType, Config c, WeightLoader w, Tokenizer t, DType workingMemoryDType,
                      DType workingMemoryQType, Optional<DType> modelQType) {
        super(inferenceType, c, w, t, workingMemoryDType, workingMemoryQType, modelQType);
    }
}