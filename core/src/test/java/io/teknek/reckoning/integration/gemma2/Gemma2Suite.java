package io.teknek.reckoning.integration.gemma2;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.AutoModelForCausaLm;
import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import io.teknek.reckoning.tensor.KvBufferCacheSettings;
import org.junit.platform.suite.api.AfterSuite;
import org.junit.platform.suite.api.BeforeSuite;
import org.junit.platform.suite.api.SelectClasses;
import org.junit.platform.suite.api.Suite;

@Suite
@SelectClasses({Gemma2PromptIT.class})
public class Gemma2Suite {

    private static volatile AbstractModel model;
    private static volatile AutoModelForCausaLm.Builder builder;

    public static AbstractModel getOrCreate() {
        if (model == null) {
            ModelFetcher fetch = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
            builder = AutoModelForCausaLm.newBuilder(fetch)
                    .withOutputHeadQuantization(DType.Q4);
            model = builder.withKvBufferCacheSettings(new KvBufferCacheSettings()
                            .withBlockSize(8))
                    .buildLocalTransformerModel();
        }
        return model;
    }

    public static AutoModelForCausaLm.Builder getBuilder() {
        return builder;
    }

    @BeforeSuite
    public static void beforeSuite() {
        getOrCreate();
    }

    @AfterSuite
    public static void afterSuite() {
        if (model != null) {
            model.close();
            model = null;
        }
    }
}
