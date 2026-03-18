package net.deliverance.distributed;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.*;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;

import java.io.File;
import java.util.Optional;
import java.util.function.Function;

public class Worker {

    public Worker(){
        ModelFetcher modelFetcher = new ModelFetcher("a", "b");
        RegisterResponse rr = new RegisterResponse();
        Function<Config, DistributedContext> configBuilder = x -> DistributedContext.builder(x)
                .setModelShard(rr.getModelShard())
                .setNumModelShards(rr.getNumModelShards())
                .setLayerShard(rr.getLayerShard())
                .setNumLayerShards(rr.getNumLayerShards())
                .build();

        Function<File, WeightLoader> weightLoaderFunction = new Function<File, WeightLoader>() {
            @Override
            public WeightLoader apply(File file) {
                return new DefaultWeightLoader(file);
            }
        };
        AbstractModel m = ModelSupport.loadModel(AbstractModel.InferenceType.FORWARD_PASS,
                modelFetcher,
                DType.F32,
                DType.F32,
                Optional.of(DType.F32),
                Optional.of(42),
                Optional.of(configBuilder),
                weightLoaderFunction);

    }
}
