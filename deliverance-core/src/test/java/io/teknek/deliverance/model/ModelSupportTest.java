package io.teknek.deliverance.model;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.fetch.ModelFetcher;
import io.teknek.deliverance.model.llama.LlamaTokenizer;
import io.teknek.deliverance.tensor.*;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ModelSupportTest {

    @Test
    void load(){
        String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        KvBufferCache.KvBuffer kvBuffer;
        try (AbstractModel z = ModelSupport.loadModel(f, DType.F32, DType.F32)) {
            assertEquals(z.tokenizer.getClass(), LlamaTokenizer.class);
            kvBuffer = z.kvBufferCache.getEphemeralKvBuffer();

        }

        assertEquals(0, kvBuffer.getCurrentContextPosition());
        kvBuffer.incrementContextPosition();
        assertEquals(1, kvBuffer.getCurrentContextPosition());
    }

    @Test
    void maybeQuantizeTest(){
        String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        KvBufferCache.KvBuffer kvBuffer;
        try (AbstractModel z = ModelSupport.loadModel(f, DType.F32, DType.F32)) {
            TensorShape ts = TensorShape.of(10,10);
            BFloat16BufferTensor bf = new BFloat16BufferTensor(ts);
            //AbstractTensor z1 = z.maybeQuantize(bf);
            //Assertions.assertEquals(0 ,  z1.get(0, 0));
        }

    }


}
