package io.teknek.deliverance.integration;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import org.junit.platform.suite.api.AfterSuite;
import org.junit.platform.suite.api.BeforeSuite;
import org.junit.platform.suite.api.SelectClasses;
import org.junit.platform.suite.api.Suite;

@Suite
@SelectClasses({Llama32ThreeBPromptIT.class})
public class Llama32ThreeBSuite {

    private static volatile AbstractModel model;
    private static volatile AutoModelForCausaLm.Builder builder;

    public static AbstractModel getOrCreate() {
        if (model == null) {
            ModelFetcher fetch = new ModelFetcher("tjake", "Llama-3.2-3B-Instruct-JQ4");
            KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                    .withMaxPrefixTokensPerPrompt(512)
                    .withMaxEntries(10_000)
                    .withBlockSize(8);
            builder = AutoModelForCausaLm.newBuilder(fetch)
                    .withWorkingQuantType(DType.I8)
                    .withKvBufferCacheSettings(settings);
            model = builder.buildLocalTransformerModel();
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
