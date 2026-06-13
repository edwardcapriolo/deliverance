package io.teknek.deliverance.integration;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import org.junit.platform.suite.api.AfterSuite;
import org.junit.platform.suite.api.BeforeSuite;
import org.junit.platform.suite.api.SelectClasses;
import org.junit.platform.suite.api.Suite;

@Suite
@SelectClasses({GemmaPromptIT.class})
public class Gemma2Suite {

    private static volatile AbstractModel model;
    private static volatile AutoModelForCausaLm.Builder builder;

    public static AbstractModel getOrCreate(){
        if (model == null){
            ModelFetcher fetch = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
            //ModelFetcher fetch = new ModelFetcher("google", "gemma-2-2b-it" );
            builder = AutoModelForCausaLm.newBuilder(fetch);
            model = builder.withKvBufferCacheSettings(new KvBufferCacheSettings(true).withBlockSize(8))
                    .buildLocalTransformerModel();
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
