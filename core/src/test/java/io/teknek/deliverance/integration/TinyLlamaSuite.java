package io.teknek.deliverance.integration;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.DirectPromptTest;
import io.teknek.deliverance.tokenizer.LlamaTokenizerTest;
import org.junit.platform.suite.api.AfterSuite;
import org.junit.platform.suite.api.BeforeSuite;
import org.junit.platform.suite.api.SelectClasses;
import org.junit.platform.suite.api.Suite;

@Suite
@SelectClasses({DirectPromptTest.class, LlamaTokenizerTest.class})
public class TinyLlamaSuite {

    private static volatile AbstractModel model;
    private static volatile AutoModelForCausaLm.Builder builder;

    public static AbstractModel getOrCreate(){
        if (model == null){
            String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
            String modelOwner = "tjake";
            ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
            builder = AutoModelForCausaLm.newBuilder(fetch);
            model = builder.build();
        }
        return model;
    }

    public static AutoModelForCausaLm.Builder getBuilder(){
        return builder;
    }

    @BeforeSuite
    public static void beforeSuite(){
       getOrCreate();
    }

    @AfterSuite
    public static void afterSuite(){
        if (model != null){
            model.close();
        }
    }
}
