package io.teknek.deliverance.integration;

import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.platform.suite.api.AfterSuite;
import org.junit.platform.suite.api.BeforeSuite;
import org.junit.platform.suite.api.SelectClasses;
import org.junit.platform.suite.api.Suite;

@Suite
@SelectClasses({Gemma4PromptIT.class})
public class Gemma4Suite {
    private static volatile AbstractModel model;
    private static volatile AutoModelForCausaLm.Builder builder;

    public static AbstractModel getOrCreate() {
        if (model == null) {
            //ModelFetcher fetch = new ModelFetcher("google", "gemma-4-E2B-it");
            ModelFetcher fetch = new ModelFetcher("edward", "gemma-4-E2B-it-JQ4");
            builder = AutoModelForCausaLm.newBuilder(fetch);
            builder.withTensorProvider(new ConfigurableTensorProvider(builder.getAllocator(),
                  new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())));
            model = builder.build();
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
        }
    }
}
